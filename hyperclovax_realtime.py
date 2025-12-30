#!/usr/bin/env python3
"""
HyperCLOVAX 실시간 AI
- 항상 마이크 청취
- 도구 실행
- 자가 학습
- INT4 + CPU Offload 극한 최적화
"""

import os
import sys
import gc
import json
import time
import queue
import threading
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")

# 경로 설정
MODEL_PATH = "/mnt/data/HyperCLOVAX/model"
sys.path.insert(0, MODEL_PATH)

MEMORY_FILE = "/mnt/data/HyperCLOVAX-AGI/ai_memory.json"


@dataclass
class AIMemory:
    """AI 메모리"""
    conversations: List[Dict] = field(default_factory=list)
    learnings: List[Dict] = field(default_factory=list)

    def save(self):
        data = {
            "conversations": self.conversations[-20:],
            "learnings": self.learnings[-100:],
            "updated": datetime.now().isoformat(),
        }
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.conversations = data.get("conversations", [])
                self.learnings = data.get("learnings", [])
            except:
                pass

    def add(self, role: str, content: str):
        self.conversations.append({
            "role": role,
            "content": content,
            "time": datetime.now().isoformat()
        })

    def learn(self, content: str):
        self.learnings.append({
            "content": content,
            "time": datetime.now().isoformat()
        })
        print(f"💡 학습: {content}")


class ToolExecutor:
    """도구 실행기"""

    def run_bash(self, cmd: str) -> str:
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            return (r.stdout + r.stderr)[:2000] or "[완료]"
        except:
            return "[오류]"

    def run_python(self, code: str) -> str:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                path = f.name
            r = subprocess.run(['python3', path], capture_output=True, text=True, timeout=30)
            os.unlink(path)
            return (r.stdout + r.stderr)[:2000] or "[완료]"
        except:
            return "[오류]"

    def read_file(self, path: str) -> str:
        try:
            with open(path, 'r') as f:
                return f.read()[:3000]
        except:
            return "[오류]"

    def write_file(self, path: str, content: str) -> str:
        try:
            with open(path, 'w') as f:
                f.write(content)
            return f"[저장: {path}]"
        except:
            return "[오류]"


class MicListener:
    """마이크 청취"""

    def __init__(self, callback):
        self.callback = callback
        self.running = False
        self.thread = None

        try:
            import speech_recognition as sr
            self.sr = sr
            self.recognizer = sr.Recognizer()
            self.available = True
        except:
            self.available = False

    def start(self):
        if not self.available:
            print("⚠️ speech_recognition 미설치")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("🎤 마이크 청취 시작")

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                with self.sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                try:
                    text = self.recognizer.recognize_google(audio, language="ko-KR")
                    if text:
                        print(f"\n🎤 [{text}]")
                        self.callback(text)
                except:
                    pass
            except:
                time.sleep(0.5)


class HyperCLOVAXRealtime:
    """HyperCLOVAX 실시간 AI"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.memory = AIMemory()
        self.memory.load()
        self.tools = ToolExecutor()
        self.input_queue = queue.Queue()
        self.mic = None
        self.running = False

    def load_model(self) -> bool:
        """모델 로드 (극한 최적화)"""
        print("\n" + "=" * 50)
        print("🚀 HyperCLOVAX 로딩 (INT4 + CPU Offload)")
        print("=" * 50)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            # CUDA 확인
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"🖥️ GPU: {gpu_name} ({vram:.1f}GB)")
            else:
                print("⚠️ CUDA 없음 - CPU 전용")

            # 토크나이저
            print("📝 토크나이저 로딩...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH, trust_remote_code=True
            )

            # INT4 양자화
            print("⚙️ INT4 양자화 설정...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            # CPU-GPU 하이브리드 device_map
            print("⚙️ CPU-GPU 하이브리드 구성...")

            # 극한 최적화: 거의 모든 것을 CPU에
            device_map = {
                # 인코더 전부 CPU
                "model.vision_model": "cpu",
                "model.audio_model": "cpu",
                "model.video_audio_compressor": "cpu",

                # 디코더 전부 CPU
                "model.discrete_vision_model": "cpu",
                "model.discrete_audio_model": "cpu",

                # Projector - GPU (작음)
                "model.mm_projector": "cuda:0",
                "model.audio_projector": "cuda:0",

                # LLM - 대부분 CPU, 마지막만 GPU
                "model.language_model.model.embed_tokens": "cpu",
                "model.language_model.model.norm": "cuda:0",
                "model.language_model.lm_head": "cuda:0",
            }

            # LLM 레이어: 32-35만 GPU, 나머지 CPU
            for i in range(36):
                layer_key = f"model.language_model.model.layers.{i}"
                if i >= 32:  # 마지막 4개 레이어만 GPU
                    device_map[layer_key] = "cuda:0"
                else:
                    device_map[layer_key] = "cpu"

            # 모델 로드
            print("🔄 모델 로딩... (수 분 소요)")

            os.makedirs("/tmp/hcx_offload", exist_ok=True)

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                quantization_config=quant_config,
                device_map=device_map,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                offload_folder="/tmp/hcx_offload",
            )

            # VRAM 상태
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"📊 VRAM: {used:.2f}GB / {total:.2f}GB")

            print("✅ 모델 로드 완료!")
            return True

        except Exception as e:
            print(f"❌ 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """텍스트 생성"""
        if self.model is None:
            return "[모델 없음]"

        try:
            import torch

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            # CPU에 있는 임베딩으로 이동
            device = "cpu"  # embed_tokens가 CPU에 있음
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )

            return response.strip()

        except Exception as e:
            return f"[생성 오류: {e}]"

    def process_tools(self, response: str) -> str:
        """도구 실행 처리"""
        import re

        # [BASH: cmd] 패턴
        def run_bash(m):
            result = self.tools.run_bash(m.group(1))
            return f"\n📌 실행 결과:\n{result}\n"
        response = re.sub(r'\[BASH:\s*(.+?)\]', run_bash, response)

        # [PYTHON: code] 패턴
        def run_python(m):
            result = self.tools.run_python(m.group(1))
            return f"\n📌 실행 결과:\n{result}\n"
        response = re.sub(r'\[PYTHON:\s*(.+?)\]', run_python, response, flags=re.DOTALL)

        # [LEARN: content] 패턴
        def learn(m):
            self.memory.learn(m.group(1))
            return ""
        response = re.sub(r'\[LEARN:\s*(.+?)\]', learn, response)

        return response

    def chat(self, user_input: str) -> str:
        """대화"""
        self.memory.add("user", user_input)

        # 컨텍스트 구성
        context = "\n".join([
            f"{'User' if c['role']=='user' else 'AI'}: {c['content']}"
            for c in self.memory.conversations[-5:]
        ])

        # 학습 내용
        learnings = ""
        if self.memory.learnings:
            learnings = "\n[학습된 지식]\n" + "\n".join([
                f"- {l['content']}" for l in self.memory.learnings[-5:]
            ])

        prompt = f"""당신은 HyperCLOVAX AI입니다. 한국어로 답변하세요.

도구 사용: [BASH: 명령], [PYTHON: 코드], [LEARN: 학습내용]

{learnings}

{context}
User: {user_input}
AI:"""

        print("\n🤖 ", end="", flush=True)
        response = self.generate(prompt)
        print(response)

        # 도구 실행
        processed = self.process_tools(response)
        if processed != response:
            print(processed)

        self.memory.add("assistant", response)
        self.memory.save()

        return response

    def on_voice(self, text: str):
        """음성 입력"""
        self.input_queue.put(text)

    def run(self, enable_mic: bool = True):
        """실행"""
        if not self.load_model():
            return

        print("\n" + "=" * 50)
        print("💬 HyperCLOVAX 대화 모드")
        print("=" * 50)
        print("명령: /quit, /learn <내용>, /status")
        print("=" * 50 + "\n")

        # 마이크
        if enable_mic:
            self.mic = MicListener(self.on_voice)
            self.mic.start()

        self.running = True

        while self.running:
            try:
                # 음성 입력 확인
                try:
                    text = self.input_queue.get_nowait()
                    self.chat(text)
                    continue
                except queue.Empty:
                    pass

                # 텍스트 입력
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                if user_input == "/quit":
                    break
                elif user_input.startswith("/learn "):
                    self.memory.learn(user_input[7:])
                elif user_input == "/status":
                    import torch
                    if torch.cuda.is_available():
                        used = torch.cuda.memory_allocated() / 1e9
                        print(f"📊 VRAM: {used:.2f}GB")
                    print(f"📚 학습: {len(self.memory.learnings)}개")
                else:
                    self.chat(user_input)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ {e}")

        print("👋 종료")
        if self.mic:
            self.mic.stop()
        self.memory.save()


def main():
    ai = HyperCLOVAXRealtime()
    ai.run(enable_mic=True)


if __name__ == "__main__":
    main()
