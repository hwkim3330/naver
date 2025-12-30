# HyperCLOVAX-SEED-Omni-8B 신경망 구조 분석

Naver HyperCLOVA X SEED 8B Omni 모델의 상세 신경망 구조 분석

## 개요

- **모델명**: HyperCLOVAX-SEED-Omni-8B
- **개발사**: Naver
- **아키텍처**: Transformer 기반 Any-to-Any 멀티모달
- **파라미터**: ~10.7B (42.97 GB @ fp32)
- **컨텍스트**: 32K tokens
- **Knowledge Cutoff**: 2025년 5월

## 지원 모달리티

| 입력 | 출력 |
|------|------|
| Text | Text |
| Image | Image |
| Video | Audio |
| Audio | - |

## 신경망 구조

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     HyperCLOVAX-SEED-Omni-8B                             │
│                     Total: 42.97 GB (10.7B params)                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                          ┌─────────────────┐                             │
│                          │   Text Input    │                             │
│                          └────────┬────────┘                             │
│                                   │                                      │
│  ┌────────────────┐  ┌────────────┴───────────┐  ┌────────────────┐     │
│  │  Vision Model  │  │                        │  │  Audio Model   │     │
│  │  (Qwen2.5-VL)  │  │                        │  │ (Qwen2-Audio)  │     │
│  │                │  │                        │  │                │     │
│  │  32 Layers     │  │                        │  │  32 Layers     │     │
│  │  390 weights   │  │                        │  │  487 weights   │     │
│  │  1280 hidden   │  │                        │  │  1280 d_model  │     │
│  │  16 heads      │  │                        │  │  20 heads      │     │
│  └───────┬────────┘  │                        │  └───────┬────────┘     │
│          │           │                        │          │              │
│          ▼           │                        │          ▼              │
│  ┌────────────────┐  │                        │  ┌────────────────┐     │
│  │  MM Projector  │  │                        │  │Audio Projector │     │
│  │  (Linear)      │  │                        │  │  (MLP)         │     │
│  │  2 weights     │  │                        │  │  4 weights     │     │
│  └───────┬────────┘  │                        │  └───────┬────────┘     │
│          │           │                        │          │              │
│          │           │    ┌──────────────┐    │          │              │
│          │           │    │  MambaMIA    │    │          │              │
│          │           │    │  Compressor  │◀───┼──────────┤              │
│          │           │    │              │    │   (Video+Audio)        │
│          │           │    │  1 Layer     │    │                        │
│          │           │    │  21 weights  │    │                        │
│          │           │    │  Mamba2 SSM  │    │                        │
│          │           │    └──────┬───────┘    │                        │
│          │           │           │            │                        │
│          └───────────┼───────────┼────────────┘                        │
│                      │           │                                      │
│                      ▼           ▼                                      │
│          ┌───────────────────────────────────────────┐                 │
│          │              Language Model               │                 │
│          │              (Llama-like)                 │                 │
│          │                                           │                 │
│          │  • 36 Layers                              │                 │
│          │  • 327 weights                            │                 │
│          │  • Hidden: 4096                           │                 │
│          │  • Attention Heads: 32                    │                 │
│          │  • KV Heads: 8 (GQA)                      │                 │
│          │  • FFN Intermediate: 12288                │                 │
│          │  • Vocab: 200,704 tokens                  │                 │
│          │  • RoPE theta: 5,000,000                  │                 │
│          │  • Activation: SiLU                       │                 │
│          └──────────────────┬────────────────────────┘                 │
│                             │                                          │
│              ┌──────────────┼──────────────┐                          │
│              │              │              │                          │
│              ▼              ▼              ▼                          │
│       ┌────────────┐ ┌────────────┐ ┌────────────┐                   │
│       │   Text     │ │  TA-Tok    │ │ CosyVoice2 │                   │
│       │  Output    │ │  Decoder   │ │  Decoder   │                   │
│       │            │ │            │ │            │                   │
│       │  (LM Head) │ │ 27 Layers  │ │  6 Layers  │                   │
│       │            │ │527 weights │ │102 weights │                   │
│       └────────────┘ └─────┬──────┘ └─────┬──────┘                   │
│                            │              │                          │
│                            ▼              ▼                          │
│                       ┌─────────┐   ┌─────────┐                      │
│                       │  Image  │   │  Audio  │                      │
│                       └─────────┘   └─────────┘                      │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## 컴포넌트 상세

### 1. Vision Model (Qwen2.5-VL)

이미지와 비디오 프레임을 인코딩하는 Vision Transformer

| 속성 | 값 |
|------|-----|
| Layers | 32 |
| Hidden Size | 1280 |
| Attention Heads | 16 |
| Patch Size | 14 |
| Spatial Merge | 2 |
| Window Size | 112 |
| Full Attention Blocks | 7, 15, 23, 31 |

### 2. Audio Model (Qwen2-Audio Encoder)

오디오/음성 입력을 인코딩하는 Whisper 스타일 인코더

| 속성 | 값 |
|------|-----|
| Layers | 32 |
| D_model | 1280 |
| Attention Heads | 20 |
| FFN Dim | 5120 |
| Mel Bins | 128 |
| Max Source Positions | 1500 |

### 3. Video-Audio Compressor (MambaMIA)

긴 비디오/오디오 시퀀스를 압축하는 Mamba2 기반 SSM

| 속성 | 값 |
|------|-----|
| Architecture | Mamba2 SSM |
| Layers | 1 |
| Weights | 21 |
| 용도 | 1시간 비디오도 32K 컨텍스트 내 처리 |

참고: [MambaMIA GitHub](https://github.com/naver-ai/mambamia)

### 4. Language Model (Core LLM)

Llama 아키텍처 기반 핵심 언어 모델

| 속성 | 값 |
|------|-----|
| Layers | 36 |
| Hidden Size | 4096 |
| Attention Heads | 32 |
| KV Heads | 8 (GQA) |
| Intermediate Size | 12288 |
| Vocab Size | 200,704 |
| Max Position | 8192 |
| RoPE Theta | 5,000,000 |
| Activation | SiLU |
| RMS Norm Eps | 1e-6 |

### 5. Discrete Vision Model (TA-Tok)

텍스트-정렬 토크나이저 기반 이미지 생성 디코더

| 속성 | 값 |
|------|-----|
| Layers | 27 |
| Weights | 527 |
| 기능 | Text-to-Image, Image Editing |

### 6. Discrete Audio Model (CosyVoice2)

음성 합성 디코더

| 속성 | 값 |
|------|-----|
| Layers | 6 |
| Weights | 102 |
| 기능 | Text-to-Speech, Voice Cloning |

## 특수 토큰

```python
# 입력 토큰
IMAGE_PAD = 128062         # <|IMAGE_PAD|>
VIDEO_PAD = 128063         # <|VIDEO_PAD|>
VIDEO_AUDIO_PAD = 128070   # <|VIDEO_AUDIO_PAD|>
AUDIO_PAD = 128071         # <|AUDIO_PAD|>

# Discrete 출력 토큰
DISCRETE_IMAGE_START = 128069
DISCRETE_IMAGE_UNIT_0 = 135168    # <|vision00000|> 시작
DISCRETE_AUDIO_START = 128074
DISCRETE_AUDIO_UNIT_0 = 128606    # <|audio0000|> 시작

# 종료 토큰
EOS_TOKEN = 128001
END_TOKEN = 128001
```

## 데이터 플로우

### Image Understanding
```
Image → Vision Model → MM Projector → LLM → Text
```

### Video Understanding
```
Video Frames + Audio → MambaMIA Compressor → LLM → Text
```

### Image Generation
```
Text → LLM → Discrete Vision Tokens → TA-Tok → RGB Image
```

### Speech Synthesis
```
Text → LLM → Discrete Audio Tokens → CosyVoice2 → Audio
```

## 하드웨어 요구사항

| 구성 | VRAM | 비고 |
|------|------|------|
| Vision Encoder | ~8 GB | |
| Audio Encoder | ~4 GB | Vision과 공유 가능 |
| LLM | ~16 GB | |
| Vision Decoder | ~16 GB | |
| Audio Decoder | ~4 GB | |
| **Total** | **~48 GB** | 4x A100 80GB 권장 |

## 파일 구조

```
model/
├── config.json                    # 전체 설정
├── configuration_vlm.py           # VLM Config 클래스
├── configuration_hyperclovax.py   # HyperCLOVAX Config
├── modeling_vlm.py                # VLM 모델 구현
├── modeling_hyperclovax.py        # HyperCLOVAX 모델
├── mambamia_videoaudio_compressor.py  # MambaMIA 구현
├── cosyvoice.py                   # CosyVoice2 디코더
├── ta_tok.py                      # TA-Tok 디코더
├── preprocessor.py                # 입력 전처리
├── processing_vlm.py              # VLM 처리
├── tokenizer.json                 # 토크나이저 (200K+ vocab)
├── model-00001-of-00010.safetensors  # 모델 가중치
├── ...
└── model-00010-of-00010.safetensors
```

## 참고 자료

- [HuggingFace Model](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B)
- [OmniServe](https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe)
- [MambaMIA Paper](https://github.com/naver-ai/mambamia)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)

## 라이선스

HyperCLOVA X SEED 8B Omni Model License Agreement
