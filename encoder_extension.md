# HyperCLOVAX 인코더 확장 가이드

## 1. 새 모달리티 인코더 추가

### 예시: 센서 인코더 추가

```python
# configuration_vlm.py에 추가
class HCXVisionConfig(PretrainedConfig):
    def __init__(
        self,
        # ... 기존 파라미터 ...
        sensor_config=None,                    # 새로 추가
        sensor_model_name_or_path=None,        # 새로 추가
        sensor_projector_type="mlp",           # 새로 추가
        sensor_start_id=128080,                # 새 특수 토큰
        # ...
    ):
```

```python
# modeling_vlm.py에 추가
class HCXVisionModel(HCXVisionPreTrainedModel):
    def __init__(self, config):
        # ... 기존 코드 ...

        # 새 센서 인코더 추가
        self.sensor_model = None
        if hasattr(config, "sensor_config") and config.sensor_config is not None:
            self.sensor_model = AutoModel.from_config(config.sensor_config)

            # 센서 Projector
            if config.sensor_projector_type == "mlp":
                self.sensor_projector = VLM_Mlp(
                    "mlp",
                    config.sensor_config.hidden_size,
                    hidden_features=config.sensor_config.hidden_size,
                    out_features=config.text_config.hidden_size,
                )
```

## 2. 기존 인코더 개선 (LoRA)

```python
from peft import LoraConfig, get_peft_model

# Vision Encoder에 LoRA 적용
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Attention만
    lora_dropout=0.1,
    bias="none",
)

model.vision_model = get_peft_model(model.vision_model, lora_config)
# 약 0.1%만 학습 (수백 MB)
```

## 3. Projector만 학습 (가장 효율적)

```python
# 모든 것 동결
for param in model.parameters():
    param.requires_grad = False

# Projector만 학습 가능
for param in model.mm_projector.parameters():
    param.requires_grad = True
for param in model.audio_projector.parameters():
    param.requires_grad = True

# 학습할 파라미터: 약 10-50MB
```

## 4. VRAM 효율적 활용

### 4.1 CPU-GPU 하이브리드

```python
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# 빈 모델 생성
with init_empty_weights():
    model = HCXVisionForCausalLM(config)

# 레이어별 분산
device_map = {
    # 인코더: CPU (느리지만 VRAM 절약)
    "model.vision_model": "cpu",
    "model.audio_model": "cpu",

    # Projector: GPU (빠름, 작음)
    "model.mm_projector": "cuda:0",
    "model.audio_projector": "cuda:0",

    # LLM: 분할
    "model.language_model.model.embed_tokens": "cuda:0",
    "model.language_model.model.layers.0": "cpu",
    # ... 레이어별 지정 ...
    "model.language_model.model.layers.35": "cuda:0",
    "model.language_model.lm_head": "cuda:0",

    # 디코더: CPU
    "model.discrete_vision_model": "cpu",
    "model.discrete_audio_model": "cpu",
}

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=MODEL_PATH,
    device_map=device_map,
    offload_folder="offload",
)
```

### 4.2 동적 레이어 로드

```python
class DynamicLayerLoader:
    """추론 시 필요한 레이어만 GPU로 이동"""

    def __init__(self, model, gpu_layers=4):
        self.model = model
        self.gpu_layers = gpu_layers
        self.current_gpu_layers = set()

    def load_layer_to_gpu(self, layer_idx):
        if layer_idx in self.current_gpu_layers:
            return

        # 오래된 레이어 CPU로 이동
        if len(self.current_gpu_layers) >= self.gpu_layers:
            old_idx = min(self.current_gpu_layers)
            self.model.language_model.model.layers[old_idx].to("cpu")
            self.current_gpu_layers.remove(old_idx)

        # 새 레이어 GPU로 이동
        self.model.language_model.model.layers[layer_idx].to("cuda:0")
        self.current_gpu_layers.add(layer_idx)
```

### 4.3 양자화 (INT8/INT4)

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
)
# 42GB → ~12GB (INT4)
```

## 5. 새 인코더 통합 예시: IMU 센서

```python
import torch.nn as nn

class IMUEncoder(nn.Module):
    """6축 IMU 데이터 인코더"""

    def __init__(self, input_dim=6, hidden_dim=256, output_dim=1280):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=4
        )
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, 6)
        x = x.permute(0, 2, 1)  # (batch, 6, seq_len)
        x = self.conv1d(x)      # (batch, 256, seq_len)
        x = x.permute(2, 0, 1)  # (seq_len, batch, 256)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, 256)
        x = self.proj(x)        # (batch, seq_len, 1280)
        return x

# Projector (IMU → LLM)
class IMUProjector(nn.Module):
    def __init__(self, input_dim=1280, output_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

## 6. 학습 전략

| 방법 | VRAM 필요 | 학습 시간 | 효과 |
|------|----------|----------|------|
| Projector만 | ~4GB | 빠름 | 새 모달리티 연결 |
| LoRA 인코더 | ~8GB | 중간 | 도메인 적응 |
| LoRA 전체 | ~16GB | 느림 | 종합 개선 |
| Full Fine-tune | ~80GB+ | 매우 느림 | 최대 효과 |

## 7. 추천 개선 순서

1. **Projector 학습** (새 모달리티 추가 시)
2. **인코더 LoRA** (도메인 특화 시)
3. **MambaMIA 개선** (긴 시퀀스 처리 개선)
4. **양자화 적용** (추론 최적화)
