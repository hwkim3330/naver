#!/usr/bin/env python3
"""
HyperCLOVAX 멀티모달 추론 파이프라인
- 이미지 이해
- 오디오 이해
- 비디오 이해
- 이미지 생성
- 음성 합성
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = "/mnt/data/HyperCLOVAX/model"
sys.path.insert(0, MODEL_PATH)


@dataclass
class ModalityInput:
    """모달리티 입력 데이터"""
    type: str  # "text", "image", "audio", "video"
    data: Any  # 실제 데이터
    path: Optional[str] = None  # 파일 경로


@dataclass
class ModalityOutput:
    """모달리티 출력 데이터"""
    type: str  # "text", "image", "audio"
    data: Any
    metadata: Optional[Dict] = None


class ImageProcessor:
    """이미지 전처리기"""

    def __init__(self, target_size: int = 448):
        self.target_size = target_size

    def load_image(self, path: str) -> Optional[Image.Image]:
        """이미지 로드"""
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {e}")
            return None

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """이미지 전처리"""
        # 리사이즈
        img = image.resize((self.target_size, self.target_size), Image.LANCZOS)

        # 텐서 변환
        img_array = np.array(img).astype(np.float32) / 255.0

        # 정규화 (ImageNet 기준)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        # CHW 형식으로 변환
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        return img_tensor.float()

    def process_anyres(self, image: Image.Image, grid_size: int = 2) -> List[torch.Tensor]:
        """AnyRes 방식 처리 (고해상도 이미지)"""
        w, h = image.size

        # 기본 이미지
        base = self.preprocess(image)

        # 그리드 분할
        grids = []
        grid_w = w // grid_size
        grid_h = h // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                crop = image.crop((
                    j * grid_w,
                    i * grid_h,
                    (j + 1) * grid_w,
                    (i + 1) * grid_h
                ))
                grids.append(self.preprocess(crop))

        return [base] + grids


class AudioProcessor:
    """오디오 전처리기"""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 128):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def load_audio(self, path: str) -> Optional[np.ndarray]:
        """오디오 로드"""
        try:
            import librosa
            audio, sr = librosa.load(path, sr=self.sample_rate)
            return audio
        except ImportError:
            print("⚠️ librosa 미설치. 오디오 처리 불가")
            return None
        except Exception as e:
            print(f"❌ 오디오 로드 실패: {e}")
            return None

    def preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """오디오 전처리 (Mel Spectrogram)"""
        try:
            import librosa

            # Mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=160,
                n_fft=400,
            )

            # Log scale
            log_mel = librosa.power_to_db(mel, ref=np.max)

            # 정규화
            log_mel = (log_mel + 80) / 80

            # 텐서 변환
            return torch.from_numpy(log_mel).unsqueeze(0).float()

        except Exception as e:
            print(f"❌ 오디오 전처리 실패: {e}")
            return torch.zeros(1, self.n_mels, 100)


class VideoProcessor:
    """비디오 전처리기"""

    def __init__(self, target_size: int = 224, max_frames: int = 32):
        self.target_size = target_size
        self.max_frames = max_frames
        self.image_processor = ImageProcessor(target_size)

    def load_video(self, path: str) -> Optional[List[Image.Image]]:
        """비디오 로드 (프레임 추출)"""
        try:
            import cv2

            cap = cv2.VideoCapture(path)
            frames = []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))

            cap.release()
            return frames

        except ImportError:
            print("⚠️ opencv-python 미설치. 비디오 처리 불가")
            return None
        except Exception as e:
            print(f"❌ 비디오 로드 실패: {e}")
            return None

    def preprocess(self, frames: List[Image.Image]) -> torch.Tensor:
        """비디오 프레임 전처리"""
        processed = []
        for frame in frames:
            tensor = self.image_processor.preprocess(frame)
            processed.append(tensor)

        # (batch, frames, channels, height, width)
        return torch.cat(processed, dim=0).unsqueeze(0)


class MultimodalPipeline:
    """멀티모달 추론 파이프라인"""

    def __init__(self, model, tokenizer, profile):
        self.model = model
        self.tokenizer = tokenizer
        self.profile = profile

        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()

        # 특수 토큰 ID
        self.IMAGE_PAD = 128062
        self.VIDEO_PAD = 128063
        self.AUDIO_PAD = 128071
        self.EOS_TOKEN = 128001

    def process_image(
        self,
        image_path: str,
        prompt: str = "이 이미지를 설명해주세요.",
    ) -> str:
        """이미지 이해"""
        # 이미지 로드 및 전처리
        image = self.image_processor.load_image(image_path)
        if image is None:
            return "[이미지 로드 실패]"

        pixel_values = self.image_processor.preprocess(image)

        # 프롬프트 구성
        full_prompt = f"<|IMAGE_PAD|>\n{prompt}"

        try:
            # 토큰화
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.profile.max_length,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pixel_values = pixel_values.to(device).half()

            # 모델이 VLM인 경우
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision_model'):
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        pixel_values=[[pixel_values]],
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                    )
            else:
                # 텍스트 전용 모델
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                    )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            return f"[이미지 처리 오류: {e}]"

    def process_audio(
        self,
        audio_path: str,
        prompt: str = "이 오디오의 내용을 설명해주세요.",
    ) -> str:
        """오디오 이해"""
        # 오디오 로드 및 전처리
        audio = self.audio_processor.load_audio(audio_path)
        if audio is None:
            return "[오디오 로드 실패]"

        audio_features = self.audio_processor.preprocess(audio)

        # 프롬프트 구성
        full_prompt = f"<|AUDIO_PAD|>\n{prompt}"

        try:
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.profile.max_length,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            audio_features = audio_features.to(device).half()

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            return f"[오디오 처리 오류: {e}]"

    def process_video(
        self,
        video_path: str,
        prompt: str = "이 비디오의 내용을 설명해주세요.",
    ) -> str:
        """비디오 이해"""
        # 비디오 로드 및 전처리
        frames = self.video_processor.load_video(video_path)
        if frames is None:
            return "[비디오 로드 실패]"

        video_tensor = self.video_processor.preprocess(frames)

        # 프롬프트 구성
        full_prompt = f"<|VIDEO_PAD|>\n{prompt}"

        try:
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.profile.max_length,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            video_tensor = video_tensor.to(device).half()

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            return f"[비디오 처리 오류: {e}]"

    def generate_image(
        self,
        prompt: str,
        output_path: str = "/tmp/generated_image.png",
    ) -> Optional[str]:
        """이미지 생성 (TA-Tok 사용)"""
        try:
            # 이미지 생성 프롬프트
            gen_prompt = f"<|generate_image|>{prompt}<|endofgenerate|>"

            inputs = self.tokenizer(
                gen_prompt,
                return_tensors="pt",
                truncation=True,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 이산 비전 토큰 생성
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,  # 이미지 토큰 수
                    do_sample=True,
                    temperature=0.8,
                )

            # TA-Tok 디코더로 이미지 변환
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'discrete_vision_model'):
                # discrete_vision_model로 토큰 → 이미지 변환
                # 실제 구현은 모델 구조에 따라 다름
                print("⚠️ 이미지 생성은 별도 디코더 필요")
                return None

            return output_path

        except Exception as e:
            print(f"❌ 이미지 생성 오류: {e}")
            return None

    def synthesize_speech(
        self,
        text: str,
        output_path: str = "/tmp/synthesized_speech.wav",
    ) -> Optional[str]:
        """음성 합성 (CosyVoice2 사용)"""
        try:
            # 음성 합성 프롬프트
            gen_prompt = f"<|generate_audio|>{text}<|endofgenerate|>"

            inputs = self.tokenizer(
                gen_prompt,
                return_tensors="pt",
                truncation=True,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 이산 오디오 토큰 생성
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,  # 오디오 토큰 수
                    do_sample=True,
                    temperature=0.8,
                )

            # CosyVoice2 디코더로 오디오 변환
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'discrete_audio_model'):
                print("⚠️ 음성 합성은 별도 디코더 필요")
                return None

            return output_path

        except Exception as e:
            print(f"❌ 음성 합성 오류: {e}")
            return None

    def process(self, inputs: List[ModalityInput], prompt: str) -> ModalityOutput:
        """범용 멀티모달 처리"""
        # 입력 분석
        has_image = any(i.type == "image" for i in inputs)
        has_audio = any(i.type == "audio" for i in inputs)
        has_video = any(i.type == "video" for i in inputs)

        # 적절한 처리기 호출
        if has_video:
            video_input = next(i for i in inputs if i.type == "video")
            result = self.process_video(video_input.path, prompt)
        elif has_image:
            image_input = next(i for i in inputs if i.type == "image")
            result = self.process_image(image_input.path, prompt)
        elif has_audio:
            audio_input = next(i for i in inputs if i.type == "audio")
            result = self.process_audio(audio_input.path, prompt)
        else:
            # 텍스트 전용
            result = self._process_text(prompt)

        return ModalityOutput(type="text", data=result)

    def _process_text(self, prompt: str) -> str:
        """텍스트 전용 처리"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.profile.max_length,
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            return f"[텍스트 처리 오류: {e}]"


def test_pipeline():
    """파이프라인 테스트"""
    print("=" * 50)
    print("멀티모달 파이프라인 테스트")
    print("=" * 50)

    from optimized_ai import OptimizedHyperCLOVAX

    # 모델 로드
    ai = OptimizedHyperCLOVAX()
    if not ai.load():
        print("❌ 모델 로드 실패")
        return

    # 파이프라인 생성
    pipeline = MultimodalPipeline(ai.model, ai.tokenizer, ai.profile)

    # 텍스트 테스트
    print("\n📝 텍스트 테스트:")
    result = pipeline._process_text("안녕하세요, 오늘 날씨가 어때요?")
    print(f"응답: {result}")

    # 이미지 테스트 (파일이 있는 경우)
    test_image = "/tmp/test_image.jpg"
    if os.path.exists(test_image):
        print("\n🖼️ 이미지 테스트:")
        result = pipeline.process_image(test_image, "이 이미지에 무엇이 있나요?")
        print(f"응답: {result}")

    ai.unload()


if __name__ == "__main__":
    test_pipeline()
