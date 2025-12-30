# HyperCLOVAX-SEED-Omni-8B 신경망 구조 분석

**Naver HyperCLOVA X SEED 8B Omni** - 한국어 최초 Any-to-Any 멀티모달 모델

## 정확한 신경망 구조

```
총 파라미터: 42.97 GB (약 10.7B params @ fp32)

┌─────────────────────────────────────────────────────────────────────────────┐
│                    HyperCLOVAX-SEED-Omni-8B Architecture                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        INPUT ENCODERS                                │    │
│  │                                                                      │    │
│  │  ┌────────────────────┐    ┌────────────────────┐                   │    │
│  │  │    Vision Model    │    │    Audio Model     │                   │    │
│  │  │   (Qwen2.5-VL)     │    │  (Qwen2-Audio)     │                   │    │
│  │  │                    │    │                    │                   │    │
│  │  │  • 32 Layers       │    │  • 32 Layers       │                   │    │
│  │  │  • 390 weights     │    │  • 487 weights     │                   │    │
│  │  │  • 1280 hidden     │    │  • 1280 d_model    │                   │    │
│  │  │  • 16 heads        │    │  • 20 heads        │                   │    │
│  │  │  • patch=14        │    │  • 128 mel bins    │                   │    │
│  │  └─────────┬──────────┘    └─────────┬──────────┘                   │    │
│  │            │                         │                              │    │
│  │            ▼                         ▼                              │    │
│  │  ┌────────────────────┐    ┌────────────────────┐                   │    │
│  │  │   MM Projector     │    │  Audio Projector   │                   │    │
│  │  │   (Linear)         │    │   (MLP)            │                   │    │
│  │  │   2 weights        │    │   4 weights        │                   │    │
│  │  └─────────┬──────────┘    └─────────┬──────────┘                   │    │
│  │            │                         │                              │    │
│  │            └────────────┬────────────┘                              │    │
│  │                         │                                           │    │
│  │                         ▼                                           │    │
│  │            ┌────────────────────────────┐                           │    │
│  │            │  Video-Audio Compressor    │  ◀── MambaMIA (Mamba2)    │    │
│  │            │  • 1 Layer (21 weights)    │      SSM 기반 압축        │    │
│  │            │  • 긴 비디오/오디오 압축   │                           │    │
│  │            └────────────┬───────────────┘                           │    │
│  │                         │                                           │    │
│  └─────────────────────────┼───────────────────────────────────────────┘    │
│                            │                                                 │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      LANGUAGE MODEL (LLM)                            │    │
│  │                                                                      │    │
│  │  • Architecture: Llama-like Transformer                             │    │
│  │  • 36 Layers                                                        │    │
│  │  • 327 weights                                                      │    │
│  │  • Hidden: 4096                                                     │    │
│  │  • Heads: 32 (KV Heads: 8 - GQA)                                   │    │
│  │  • FFN: 12288 (intermediate)                                       │    │
│  │  • Vocab: 200,704 tokens                                           │    │
│  │  • Context: 32K (max_position_embeddings: 8192, rope_theta: 5M)    │    │
│  │  • Activation: SiLU                                                │    │
│  │                                                                      │    │
│  └─────────────────────────┬───────────────────────────────────────────┘    │
│                            │                                                 │
│                            ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      OUTPUT DECODERS                                 │    │
│  │                                                                      │    │
│  │  ┌────────────────────────┐    ┌────────────────────────┐           │    │
│  │  │  Discrete Vision       │    │  Discrete Audio        │           │    │
│  │  │  (TA-Tok)              │    │  (CosyVoice2)          │           │    │
│  │  │                        │    │                        │           │    │
│  │  │  • 27 Layers           │    │  • 6 Layers            │           │    │
│  │  │  • 527 weights         │    │  • 102 weights         │           │    │
│  │  │  • Text-to-Image       │    │  • Text-to-Speech      │           │    │
│  │  │  • Image Editing       │    │  • Voice Cloning       │           │    │
│  │  └────────────────────────┘    └────────────────────────┘           │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 컴포넌트별 상세

| Component | Weights | Layers | 기반 아키텍처 | 역할 |
|-----------|---------|--------|--------------|------|
| `vision_model` | 390 | 32 | Qwen2.5-VL ViT | 이미지/비디오 인코딩 |
| `audio_model` | 487 | 32 | Qwen2-Audio Encoder | 오디오/음성 인코딩 |
| `video_audio_compressor` | 21 | 1 | MambaMIA (Mamba2 SSM) | 긴 시퀀스 압축 |
| `mm_projector` | 2 | - | Linear | Vision→LLM 투영 |
| `audio_projector` | 4 | - | MLP | Audio→LLM 투영 |
| `language_model` | 327 | 36 | Llama-like | 핵심 언어 모델 |
| `discrete_vision_model` | 527 | 27 | TA-Tok | 이미지 생성/편집 |
| `discrete_audio_model` | 102 | 6 | CosyVoice2 | 음성 합성 |

## 특수 토큰 ID

```python
# 입력 토큰
IMAGE_PAD = 128062      # <|IMAGE_PAD|>
VIDEO_PAD = 128063      # <|VIDEO_PAD|>
AUDIO_PAD = 128071      # <|AUDIO_PAD|>
VIDEO_AUDIO_PAD = 128070  # <|VIDEO_AUDIO_PAD|>

# 출력 (Discrete) 토큰
DISCRETE_IMAGE_START = 128069   # 이미지 생성 시작
DISCRETE_IMAGE_UNIT_0 = 135168  # 이미지 토큰 시작
DISCRETE_AUDIO_START = 128074   # 오디오 생성 시작
DISCRETE_AUDIO_UNIT_0 = 128606  # 오디오 토큰 시작
```

## 데이터 플로우

### 1. 이미지 이해 (Image → Text)
```
Image → Vision Model (Qwen2.5-VL) → MM Projector → LLM → Text
```

### 2. 비디오 이해 (Video + Audio → Text)
```
Video Frames ─┐
              ├→ Video-Audio Compressor (MambaMIA) → LLM → Text
Audio ────────┘
```

### 3. 이미지 생성 (Text → Image)
```
Text → LLM → Discrete Vision Tokens → TA-Tok Decoder → Image
```

### 4. 음성 합성 (Text → Audio)
```
Text → LLM → Discrete Audio Tokens → CosyVoice2 Decoder → Audio
```

## 핵심 혁신

### MambaMIA (Video-Audio Compressor)
- **Mamba2 SSM** 기반 시퀀스 압축
- 긴 비디오/오디오의 토큰 폭발 방지
- 시간 1시간 비디오도 32K 컨텍스트 내 처리 가능

### TA-Tok (Text-Aligned Tokenizer)
- 텍스트와 정렬된 이미지 토큰화
- Discrete image tokens로 이미지 생성
- 편집도 동일 메커니즘으로 가능

### CosyVoice2
- 음성 클로닝 지원
- Discrete audio tokens 사용
- 한국어 TTS 최적화

## 하드웨어 요구사항

| 구성요소 | VRAM |
|----------|------|
| Vision Encoder | ~8 GB |
| Audio Encoder | ~4 GB |
| LLM (8B) | ~16 GB |
| Vision Decoder | ~16 GB |
| Audio Decoder | ~4 GB |
| **합계** | **~48 GB** (3-4x A100) |

## 참고 자료

- [HuggingFace Model](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B)
- [OmniServe](https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe)
- [MambaMIA](https://github.com/naver-ai/mambamia)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)

## 라이선스

HyperCLOVA X SEED 8B Omni Model License Agreement
