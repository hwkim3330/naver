#!/bin/bash
# HyperCLOVAX Optimized AI 실행 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="/mnt/data/HyperCLOVAX/model"

echo "=============================================="
echo "  HyperCLOVAX Optimized AI"
echo "=============================================="

# 모델 확인
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 모델 경로 없음: $MODEL_PATH"
    echo "   HuggingFace에서 모델 다운로드:"
    echo "   huggingface-cli download naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B --local-dir $MODEL_PATH"
    exit 1
fi

# GPU 확인
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️ GPU 정보:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# 실행 모드 선택
case "${1:-chat}" in
    chat)
        echo "💬 대화 모드 실행"
        python3 "$SCRIPT_DIR/optimized_ai.py"
        ;;
    test)
        echo "🧪 테스트 모드 실행"
        python3 "$SCRIPT_DIR/multimodal_pipeline.py"
        ;;
    *)
        echo "사용법: $0 [chat|test]"
        echo "  chat - 대화 모드 (기본)"
        echo "  test - 멀티모달 테스트"
        ;;
esac
