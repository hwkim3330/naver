# HyperCLOVAX Neural AGI

**진짜 신경망 레벨 AGI** - 프롬프트 엔지니어링이 아닌 실제 신경망 내부에 접근하는 AGI 시스템

## HyperCLOVAX OMNI 구조

```
┌──────────────────────────────────────────────────────────────────┐
│                   HyperCLOVAX OMNI 8B Architecture               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐                    │
│   │ Vision   │   │  Audio   │   │  Video   │                    │
│   │ Qwen2.5  │   │  Qwen2   │   │  Frames  │                    │
│   │   VL     │   │  Audio   │   │          │                    │
│   │ (32 Layers)  │ (32 Layers)  │          │                    │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘                    │
│        │              │              │                           │
│        └──────────────┼──────────────┘                           │
│                       ▼                                          │
│              ┌────────────────┐                                  │
│              │   MambaMIA     │  ← Mamba2 SSM based              │
│              │  (Compressor)  │    Video/Audio Compression       │
│              └───────┬────────┘                                  │
│                      ▼                                           │
│              ┌────────────────┐                                  │
│              │   Projector    │                                  │
│              │    (Linear)    │                                  │
│              └───────┬────────┘                                  │
│                      ▼                                           │
│   ┌──────────────────────────────────────────────┐              │
│   │           LLM (Llama-like)                   │              │
│   │    36 layers, 4096 hidden, 32 heads          │              │
│   │    vocab: 200,704 tokens                     │              │
│   └──────────────────┬───────────────────────────┘              │
│                      │                                           │
│        ┌─────────────┴─────────────┐                            │
│        ▼                           ▼                             │
│   ┌─────────┐               ┌───────────┐                       │
│   │ TA-Tok  │               │ CosyVoice │                       │
│   │ (Image) │               │  (Audio)  │                       │
│   │ Gen     │               │  Gen      │                       │
│   └─────────┘               └───────────┘                       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## 핵심 기능

### 1. 신경망 내부 접근
- **Hidden States 모니터링**: 각 레이어의 활성화 값 실시간 분석
- **Attention 패턴 분석**: 어텐션 엔트로피로 집중도 측정
- **MambaMIA SSM States**: Mamba2 상태 공간 모델의 내부 상태 활용

### 2. 진짜 자기 인식
```python
# 일반적인 "가짜" 자기 인식
response = model.generate("너의 상태를 설명해")  # 그냥 텍스트 생성

# Neural AGI의 진짜 자기 인식
neural_state = introspection.analyze_activations()
# → 실제 뉴런 활성화 기반 상태:
#   - confidence: 0.73 (활성화 패턴 기반)
#   - focus_level: 0.81 (어텐션 엔트로피 기반)
#   - uncertainty: 0.15 (활성화 분산 기반)
```

### 3. 신경망 메모리
- **Embedding Space 기반 저장**: 텍스트를 모델의 실제 embedding으로 인코딩
- **Cosine Similarity 검색**: 의미적으로 유사한 기억 검색
- **영구 저장**: 임베딩 텐서와 텍스트 함께 저장

### 4. 재귀적 자기 개선 (LoRA)
- 모델 구조 분석
- 약점 기반 LoRA 타겟 제안
- 동적 어댑터 적용 가능

## 설치

```bash
# 의존성
pip install torch transformers einops timm

# HyperCLOVAX 모델 다운로드 (46GB)
# huggingface-cli download naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B
```

## 사용법

```bash
# 대화 모드
python neural_agi.py --mode chat

# 모델 분석
python neural_agi.py --mode analyze

# 구조 테스트 (모델 없이)
python neural_agi.py --mode test --no-model
```

### 명령어
- `/introspect` - 신경망 상태 기반 자기 성찰
- `/analyze` - 모델 구조 분석 (파라미터 수, 모듈 등)
- `/memory <query>` - 신경망 메모리 검색
- `exit` - 종료

## 기술 스택

- **Base Model**: HyperCLOVAX-SEED-Omni-8B (Naver)
- **Vision**: Qwen2.5-VL ViT (32층, 1280d)
- **Audio**: Qwen2-Audio Encoder (32층, 1280d)
- **Compressor**: MambaMIA (Mamba2 SSM)
- **LLM**: Llama-like (36층, 4096d, 8B params)
- **Output**: CosyVoice2 (음성), TA-Tok (이미지)

## 프로젝트 구조

```
HyperCLOVAX-AGI/
├── neural_agi.py       # 메인 AGI 시스템
├── README.md           # 이 파일
├── requirements.txt    # 의존성
└── data/               # 상태 저장
    ├── neural_state.json
    └── neural_memory/
        ├── memories.json
        └── embeddings.pt
```

## 라이선스

MIT License

## 참고

- [HyperCLOVAX-SEED-Omni-8B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B)
- [MambaMIA](https://github.com/naver-ai/mambamia)
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
