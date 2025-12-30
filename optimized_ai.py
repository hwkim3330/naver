#!/usr/bin/env python3
"""
HyperCLOVAX Optimized AI
- INT4/INT8 양자화
- CPU-GPU 하이브리드 로딩
- 동적 VRAM 관리
- 멀티모달 지원

환경: GTX 1050 Ti 4GB ~ RTX 4090 24GB
"""

import os
import sys
import gc
import json
import torch
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# 모델 경로
MODEL_PATH = "/mnt/data/HyperCLOVAX/model"

# 모델 파일 경로 추가
sys.path.insert(0, MODEL_PATH)


@dataclass
class HardwareProfile:
    """하드웨어 프로파일"""
    name: str
    vram_gb: float
    use_quantization: bool = True
    quantization_bits: int = 4
    cpu_offload: bool = True
    max_gpu_layers: int = 8
    batch_size: int = 1
    max_length: int = 512

    # 프리셋
    PROFILES: Dict[str, 'HardwareProfile'] = field(default_factory=dict)


# 하드웨어 프리셋
HARDWARE_PRESETS = {
    "gtx1050ti_4gb": HardwareProfile(
        name="GTX 1050 Ti",
        vram_gb=4.0,
        use_quantization=True,
        quantization_bits=4,
        cpu_offload=True,
        max_gpu_layers=4,
        batch_size=1,
        max_length=256,
    ),
    "rtx3060_12gb": HardwareProfile(
        name="RTX 3060",
        vram_gb=12.0,
        use_quantization=True,
        quantization_bits=4,
        cpu_offload=True,
        max_gpu_layers=16,
        batch_size=1,
        max_length=1024,
    ),
    "rtx3090_24gb": HardwareProfile(
        name="RTX 3090",
        vram_gb=24.0,
        use_quantization=True,
        quantization_bits=4,
        cpu_offload=False,
        max_gpu_layers=36,
        batch_size=2,
        max_length=2048,
    ),
    "a100_40gb": HardwareProfile(
        name="A100 40GB",
        vram_gb=40.0,
        use_quantization=False,
        quantization_bits=16,
        cpu_offload=False,
        max_gpu_layers=36,
        batch_size=4,
        max_length=4096,
    ),
    "cpu_only": HardwareProfile(
        name="CPU Only",
        vram_gb=0.0,
        use_quantization=True,
        quantization_bits=8,
        cpu_offload=True,
        max_gpu_layers=0,
        batch_size=1,
        max_length=256,
    ),
}


class VRAMManager:
    """동적 VRAM 관리자"""

    def __init__(self, max_vram_gb: float = 4.0):
        self.max_vram_gb = max_vram_gb
        self.max_vram_bytes = int(max_vram_gb * 1024 * 1024 * 1024)

    def get_available_vram(self) -> float:
        """사용 가능한 VRAM (GB)"""
        if not torch.cuda.is_available():
            return 0.0

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory

        available = total - reserved
        return available / (1024 ** 3)

    def get_used_vram(self) -> float:
        """사용 중인 VRAM (GB)"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)

    def clear_cache(self):
        """VRAM 캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def can_allocate(self, size_gb: float) -> bool:
        """할당 가능 여부"""
        return self.get_available_vram() >= size_gb

    def print_status(self):
        """VRAM 상태 출력"""
        if torch.cuda.is_available():
            used = self.get_used_vram()
            available = self.get_available_vram()
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"📊 VRAM: {used:.2f}GB / {total:.2f}GB (여유: {available:.2f}GB)")
        else:
            print("📊 CUDA 사용 불가 - CPU 모드")


class DynamicLayerManager:
    """동적 레이어 GPU/CPU 관리"""

    def __init__(self, model, max_gpu_layers: int = 4):
        self.model = model
        self.max_gpu_layers = max_gpu_layers
        self.current_gpu_layers = set()
        self.layer_access_count = {}

    def get_layer(self, layer_idx: int):
        """레이어 가져오기 (필요시 GPU로 이동)"""
        if not hasattr(self.model, 'language_model'):
            return None

        layers = self.model.language_model.model.layers
        if layer_idx >= len(layers):
            return None

        # 접근 횟수 기록
        self.layer_access_count[layer_idx] = self.layer_access_count.get(layer_idx, 0) + 1

        # 이미 GPU에 있으면 반환
        if layer_idx in self.current_gpu_layers:
            return layers[layer_idx]

        # GPU 공간 확보
        while len(self.current_gpu_layers) >= self.max_gpu_layers:
            self._evict_least_used()

        # GPU로 이동
        layers[layer_idx] = layers[layer_idx].to('cuda:0')
        self.current_gpu_layers.add(layer_idx)

        return layers[layer_idx]

    def _evict_least_used(self):
        """가장 적게 사용된 레이어를 CPU로 이동"""
        if not self.current_gpu_layers:
            return

        # 가장 적게 사용된 레이어 찾기
        least_used = min(
            self.current_gpu_layers,
            key=lambda x: self.layer_access_count.get(x, 0)
        )

        # CPU로 이동
        layers = self.model.language_model.model.layers
        layers[least_used] = layers[least_used].to('cpu')
        self.current_gpu_layers.remove(least_used)

        # 접근 횟수 리셋
        self.layer_access_count[least_used] = 0

    def move_all_to_cpu(self):
        """모든 레이어를 CPU로 이동"""
        if not hasattr(self.model, 'language_model'):
            return

        layers = self.model.language_model.model.layers
        for idx in list(self.current_gpu_layers):
            layers[idx] = layers[idx].to('cpu')
        self.current_gpu_layers.clear()


def detect_hardware() -> HardwareProfile:
    """하드웨어 자동 감지"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA 사용 불가 - CPU 모드")
        return HARDWARE_PRESETS["cpu_only"]

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"🖥️ GPU: {gpu_name}")
    print(f"📊 VRAM: {total_vram:.1f} GB")

    # VRAM 기반 프로파일 선택
    if total_vram < 6:
        profile = HARDWARE_PRESETS["gtx1050ti_4gb"]
    elif total_vram < 16:
        profile = HARDWARE_PRESETS["rtx3060_12gb"]
    elif total_vram < 32:
        profile = HARDWARE_PRESETS["rtx3090_24gb"]
    else:
        profile = HARDWARE_PRESETS["a100_40gb"]

    # 실제 VRAM으로 조정
    profile.vram_gb = total_vram
    print(f"✅ 프로파일: {profile.name}")

    return profile


def create_device_map(profile: HardwareProfile) -> Dict[str, str]:
    """device_map 생성"""
    device_map = {}

    if profile.cpu_offload or profile.vram_gb < 8:
        # CPU-GPU 하이브리드
        device_map = {
            # 인코더: CPU (VRAM 절약)
            "model.vision_model": "cpu",
            "model.audio_model": "cpu",

            # Projector: GPU (작고 빠름)
            "model.mm_projector": "cuda:0" if profile.vram_gb > 2 else "cpu",
            "model.audio_projector": "cuda:0" if profile.vram_gb > 2 else "cpu",

            # MambaMIA: CPU
            "model.video_audio_compressor": "cpu",

            # LLM 임베딩: GPU
            "model.language_model.model.embed_tokens": "cuda:0" if profile.vram_gb > 2 else "cpu",

            # LLM 출력: GPU
            "model.language_model.lm_head": "cuda:0" if profile.vram_gb > 2 else "cpu",

            # 디코더: CPU
            "model.discrete_vision_model": "cpu",
            "model.discrete_audio_model": "cpu",
        }

        # LLM 레이어 분배
        total_layers = 36
        gpu_layers = min(profile.max_gpu_layers, total_layers)

        for i in range(total_layers):
            layer_key = f"model.language_model.model.layers.{i}"
            if i >= total_layers - gpu_layers:
                device_map[layer_key] = "cuda:0"
            else:
                device_map[layer_key] = "cpu"

        # Norm 레이어
        device_map["model.language_model.model.norm"] = "cuda:0" if profile.vram_gb > 2 else "cpu"

    else:
        # 전체 GPU
        device_map = "auto"

    return device_map


def get_quantization_config(profile: HardwareProfile):
    """양자화 설정 생성"""
    if not profile.use_quantization:
        return None

    from transformers import BitsAndBytesConfig

    if profile.quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif profile.quantization_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )

    return None


class OptimizedHyperCLOVAX:
    """최적화된 HyperCLOVAX 모델"""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        profile: Optional[HardwareProfile] = None,
        auto_detect: bool = True,
    ):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.vram_manager = None
        self.layer_manager = None

        # 하드웨어 프로파일
        if profile is None and auto_detect:
            self.profile = detect_hardware()
        elif profile is None:
            self.profile = HARDWARE_PRESETS["cpu_only"]
        else:
            self.profile = profile

        self.vram_manager = VRAMManager(self.profile.vram_gb)

    def load(self) -> bool:
        """모델 로드"""
        print("\n" + "=" * 50)
        print("🚀 HyperCLOVAX 모델 로딩")
        print("=" * 50)

        try:
            from transformers import AutoTokenizer, AutoConfig

            # Config 로드
            print("📄 설정 로드 중...")
            config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            # Tokenizer 로드
            print("📝 토크나이저 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            # 양자화 설정
            quant_config = get_quantization_config(self.profile)

            # Device map 생성
            device_map = create_device_map(self.profile)

            print(f"⚙️ 양자화: {'INT' + str(self.profile.quantization_bits) if self.profile.use_quantization else 'FP16'}")
            print(f"⚙️ CPU Offload: {'활성화' if self.profile.cpu_offload else '비활성화'}")

            # 모델 로드
            print("🔄 모델 로드 중... (시간이 걸릴 수 있습니다)")

            from transformers import AutoModelForCausalLM

            load_kwargs = {
                "pretrained_model_name_or_path": self.model_path,
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }

            if quant_config:
                load_kwargs["quantization_config"] = quant_config

            if device_map != "auto":
                load_kwargs["device_map"] = device_map
            else:
                load_kwargs["device_map"] = "auto"

            # offload 폴더 설정
            offload_dir = "/tmp/hcx_offload"
            os.makedirs(offload_dir, exist_ok=True)
            load_kwargs["offload_folder"] = offload_dir

            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

            # 동적 레이어 관리자 설정
            if self.profile.cpu_offload:
                self.layer_manager = DynamicLayerManager(
                    self.model.model if hasattr(self.model, 'model') else self.model,
                    max_gpu_layers=self.profile.max_gpu_layers
                )

            self.vram_manager.print_status()
            print("✅ 모델 로드 완료!")

            return True

        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """텍스트 생성"""
        if self.model is None:
            return "[모델이 로드되지 않았습니다]"

        if max_new_tokens is None:
            max_new_tokens = min(256, self.profile.max_length)

        try:
            # 입력 토큰화
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.profile.max_length,
            )

            # 디바이스 이동
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 생성
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # 디코딩
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            return f"[생성 오류: {e}]"

    def chat(
        self,
        message: str,
        system_prompt: str = "당신은 HyperCLOVAX 기반의 도움이 되는 AI 어시스턴트입니다.",
        history: List[Dict[str, str]] = None,
    ) -> str:
        """대화형 생성"""
        if history is None:
            history = []

        # 대화 형식 구성
        prompt_parts = [f"System: {system_prompt}\n"]

        for turn in history:
            if turn.get("role") == "user":
                prompt_parts.append(f"User: {turn['content']}\n")
            elif turn.get("role") == "assistant":
                prompt_parts.append(f"Assistant: {turn['content']}\n")

        prompt_parts.append(f"User: {message}\nAssistant:")

        prompt = "".join(prompt_parts)

        return self.generate(prompt)

    def unload(self):
        """모델 언로드"""
        if self.layer_manager:
            self.layer_manager.move_all_to_cpu()

        del self.model
        del self.tokenizer

        self.model = None
        self.tokenizer = None

        self.vram_manager.clear_cache()
        print("🗑️ 모델 언로드 완료")


class InteractiveAI:
    """대화형 인터페이스"""

    def __init__(self, model: OptimizedHyperCLOVAX):
        self.model = model
        self.history = []
        self.system_prompt = "당신은 HyperCLOVAX 기반의 도움이 되는 AI 어시스턴트입니다. 한국어로 자연스럽게 대화합니다."

    def run(self):
        """대화 루프 실행"""
        print("\n" + "=" * 50)
        print("💬 HyperCLOVAX 대화 모드")
        print("=" * 50)
        print("명령어:")
        print("  /quit - 종료")
        print("  /clear - 대화 초기화")
        print("  /system <prompt> - 시스템 프롬프트 변경")
        print("  /status - VRAM 상태")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                # 명령어 처리
                if user_input.startswith("/"):
                    if user_input == "/quit":
                        print("👋 종료합니다.")
                        break
                    elif user_input == "/clear":
                        self.history = []
                        print("🗑️ 대화 기록 초기화됨")
                        continue
                    elif user_input.startswith("/system "):
                        self.system_prompt = user_input[8:]
                        print(f"⚙️ 시스템 프롬프트 변경됨")
                        continue
                    elif user_input == "/status":
                        self.model.vram_manager.print_status()
                        continue
                    else:
                        print("❓ 알 수 없는 명령어")
                        continue

                # 응답 생성
                print("🤖 AI: ", end="", flush=True)
                response = self.model.chat(
                    message=user_input,
                    system_prompt=self.system_prompt,
                    history=self.history,
                )
                print(response)

                # 히스토리 업데이트
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": response})

                # 히스토리 크기 제한
                if len(self.history) > 10:
                    self.history = self.history[-10:]

            except KeyboardInterrupt:
                print("\n👋 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류: {e}")


def main():
    """메인 함수"""
    print("=" * 60)
    print("  HyperCLOVAX Optimized AI")
    print("  환경 적응형 멀티모달 AI")
    print("=" * 60)

    # 모델 초기화
    ai = OptimizedHyperCLOVAX(
        model_path=MODEL_PATH,
        auto_detect=True,
    )

    # 모델 로드
    if not ai.load():
        print("❌ 모델 로드 실패. 종료합니다.")
        return

    # 대화 모드 실행
    interactive = InteractiveAI(ai)
    interactive.run()

    # 정리
    ai.unload()


if __name__ == "__main__":
    main()
