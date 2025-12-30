#!/usr/bin/env python3
"""
실시간 AI 에이전트
- 항상 마이크 청취
- 도구 실행
- 자가 학습/개선
- GTX 1050 Ti 4GB 최적화
"""

import os
import sys
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

# 설정
MODEL_NAME = "qwen2.5:3b"  # 1.9GB - GTX 1050 Ti에 적합
OLLAMA_URL = "http://localhost:11434"
MEMORY_FILE = "/mnt/data/HyperCLOVAX-AGI/ai_memory.json"
LEARNING_FILE = "/mnt/data/HyperCLOVAX-AGI/ai_learnings.json"


@dataclass
class AIMemory:
    """AI 메모리 (자가 수정 가능)"""
    short_term: List[Dict] = field(default_factory=list)  # 최근 대화
    long_term: Dict[str, Any] = field(default_factory=dict)  # 영구 지식
    learnings: List[Dict] = field(default_factory=list)  # 학습 내용
    preferences: Dict[str, Any] = field(default_factory=dict)  # 사용자 선호

    def save(self):
        """메모리 저장"""
        data = {
            "short_term": self.short_term[-20:],  # 최근 20개만
            "long_term": self.long_term,
            "learnings": self.learnings[-100:],  # 최근 100개
            "preferences": self.preferences,
            "last_updated": datetime.now().isoformat(),
        }
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        """메모리 로드"""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.short_term = data.get("short_term", [])
                self.long_term = data.get("long_term", {})
                self.learnings = data.get("learnings", [])
                self.preferences = data.get("preferences", {})
                print(f"📚 메모리 로드: {len(self.learnings)}개 학습 내용")
            except:
                pass

    def add_learning(self, category: str, content: str, source: str = "conversation"):
        """학습 내용 추가"""
        self.learnings.append({
            "category": category,
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def add_conversation(self, role: str, content: str):
        """대화 추가"""
        self.short_term.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.short_term) > 20:
            self.short_term = self.short_term[-20:]


class ToolExecutor:
    """도구 실행기"""

    def __init__(self):
        self.tools = {
            "bash": self.run_bash,
            "python": self.run_python,
            "search": self.web_search,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
            "system_info": self.system_info,
            "time": self.get_time,
            "calculator": self.calculator,
        }

    def run_bash(self, command: str) -> str:
        """Bash 명령 실행"""
        try:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True,
                timeout=30
            )
            output = result.stdout + result.stderr
            return output[:2000] if output else "[완료]"
        except subprocess.TimeoutExpired:
            return "[타임아웃]"
        except Exception as e:
            return f"[오류: {e}]"

    def run_python(self, code: str) -> str:
        """Python 코드 실행"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True, text=True,
                timeout=30
            )
            os.unlink(temp_path)
            output = result.stdout + result.stderr
            return output[:2000] if output else "[완료]"
        except Exception as e:
            return f"[오류: {e}]"

    def web_search(self, query: str) -> str:
        """웹 검색 (시뮬레이션)"""
        return f"[웹 검색 '{query}' - 실제 구현 필요]"

    def read_file(self, path: str) -> str:
        """파일 읽기"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content[:3000]
        except Exception as e:
            return f"[오류: {e}]"

    def write_file(self, path: str, content: str) -> str:
        """파일 쓰기"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"[저장됨: {path}]"
        except Exception as e:
            return f"[오류: {e}]"

    def list_files(self, path: str = ".") -> str:
        """파일 목록"""
        try:
            files = os.listdir(path)
            return "\n".join(files[:50])
        except Exception as e:
            return f"[오류: {e}]"

    def system_info(self) -> str:
        """시스템 정보"""
        try:
            import platform
            info = [
                f"OS: {platform.system()} {platform.release()}",
                f"Python: {platform.python_version()}",
                f"Machine: {platform.machine()}",
            ]

            # GPU 정보
            try:
                gpu = subprocess.run(
                    "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader",
                    shell=True, capture_output=True, text=True
                )
                if gpu.stdout:
                    info.append(f"GPU: {gpu.stdout.strip()}")
            except:
                pass

            return "\n".join(info)
        except Exception as e:
            return f"[오류: {e}]"

    def get_time(self) -> str:
        """현재 시간"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculator(self, expression: str) -> str:
        """계산기"""
        try:
            # 안전한 수식만 허용
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expression):
                return "[허용되지 않는 문자]"
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"[오류: {e}]"

    def execute(self, tool_name: str, **kwargs) -> str:
        """도구 실행"""
        if tool_name not in self.tools:
            return f"[알 수 없는 도구: {tool_name}]"

        tool = self.tools[tool_name]
        try:
            # 첫 번째 인자 추출
            if kwargs:
                first_arg = list(kwargs.values())[0]
                return tool(first_arg) if len(kwargs) == 1 else tool(**kwargs)
            return tool()
        except Exception as e:
            return f"[실행 오류: {e}]"


class MicrophoneListener:
    """마이크 청취기"""

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.running = False
        self.thread = None

        # 음성 인식 확인
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.has_sr = True
            print("✅ 음성 인식 준비됨")
        except ImportError:
            self.has_sr = False
            print("⚠️ speech_recognition 미설치 - 텍스트 입력만 사용")

    def start(self):
        """청취 시작"""
        if not self.has_sr:
            return

        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print("🎤 마이크 청취 시작")

    def stop(self):
        """청취 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _listen_loop(self):
        """청취 루프"""
        import speech_recognition as sr

        while self.running:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                try:
                    text = self.recognizer.recognize_google(audio, language="ko-KR")
                    if text:
                        print(f"\n🎤 인식: {text}")
                        self.callback(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass

            except Exception:
                time.sleep(1)


class OllamaClient:
    """Ollama 클라이언트"""

    def __init__(self, model: str = MODEL_NAME):
        self.model = model
        self.base_url = OLLAMA_URL

    def generate(self, prompt: str, system: str = None, stream: bool = True) -> str:
        """텍스트 생성"""
        import requests

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }

        if system:
            data["system"] = system

        try:
            if stream:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=data,
                    stream=True,
                    timeout=60
                )

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            text = chunk["response"]
                            print(text, end="", flush=True)
                            full_response += text
                        if chunk.get("done"):
                            break

                print()  # 줄바꿈
                return full_response
            else:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=data,
                    timeout=60
                )
                return response.json().get("response", "")

        except Exception as e:
            return f"[Ollama 오류: {e}]"

    def is_available(self) -> bool:
        """Ollama 사용 가능 여부"""
        import requests
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class RealtimeAI:
    """실시간 AI 에이전트"""

    def __init__(self):
        self.memory = AIMemory()
        self.memory.load()

        self.tools = ToolExecutor()
        self.ollama = OllamaClient()

        self.mic_listener = None
        self.input_queue = queue.Queue()
        self.running = False

        # 시스템 프롬프트
        self.system_prompt = """당신은 실시간 AI 어시스턴트입니다.

특징:
- 한국어로 자연스럽게 대화
- 도구 사용 가능 (bash, python, file 등)
- 학습 내용을 기억하고 개선

도구 사용 형식:
[TOOL: tool_name](argument)

사용 가능한 도구:
- bash: 쉘 명령 실행
- python: 파이썬 코드 실행
- read_file: 파일 읽기
- write_file: 파일 쓰기
- list_files: 파일 목록
- system_info: 시스템 정보
- time: 현재 시간
- calculator: 계산

학습 형식:
[LEARN: category](content)

예시:
"현재 시간은?" → [TOOL: time]()
"3+5 계산해줘" → [TOOL: calculator](3+5)
"홈 디렉토리 파일 보여줘" → [TOOL: bash](ls ~)
"""

    def process_response(self, response: str) -> str:
        """응답 처리 (도구 실행, 학습 저장)"""
        import re

        # 도구 실행 패턴
        tool_pattern = r'\[TOOL:\s*(\w+)\]\(([^)]*)\)'

        def execute_tool(match):
            tool_name = match.group(1)
            arg = match.group(2)
            result = self.tools.execute(tool_name, arg=arg)
            return f"\n📌 {tool_name} 결과:\n{result}\n"

        response = re.sub(tool_pattern, execute_tool, response)

        # 학습 패턴
        learn_pattern = r'\[LEARN:\s*(\w+)\]\(([^)]*)\)'

        def save_learning(match):
            category = match.group(1)
            content = match.group(2)
            self.memory.add_learning(category, content)
            return f"\n💡 학습됨: [{category}] {content}\n"

        response = re.sub(learn_pattern, save_learning, response)

        return response

    def build_prompt(self, user_input: str) -> str:
        """프롬프트 구성"""
        # 최근 대화 컨텍스트
        context_parts = []

        for msg in self.memory.short_term[-5:]:
            role = "User" if msg["role"] == "user" else "AI"
            context_parts.append(f"{role}: {msg['content']}")

        # 관련 학습 내용
        if self.memory.learnings:
            recent_learnings = self.memory.learnings[-5:]
            learning_text = "\n".join([
                f"- [{l['category']}] {l['content']}"
                for l in recent_learnings
            ])
            context_parts.append(f"\n[학습된 지식]\n{learning_text}")

        context = "\n".join(context_parts)

        prompt = f"""[대화 기록]
{context}

User: {user_input}
AI:"""

        return prompt

    def chat(self, user_input: str) -> str:
        """대화"""
        # 메모리에 추가
        self.memory.add_conversation("user", user_input)

        # 프롬프트 구성
        prompt = self.build_prompt(user_input)

        # 응답 생성
        print("\n🤖 AI: ", end="", flush=True)
        response = self.ollama.generate(prompt, system=self.system_prompt)

        # 응답 처리 (도구 실행 등)
        processed = self.process_response(response)
        if processed != response:
            print(processed)

        # 메모리에 추가
        self.memory.add_conversation("assistant", response)
        self.memory.save()

        return response

    def on_voice_input(self, text: str):
        """음성 입력 처리"""
        self.input_queue.put(text)

    def run(self, enable_mic: bool = True):
        """실행"""
        print("\n" + "=" * 50)
        print("  🤖 실시간 AI 에이전트")
        print("=" * 50)

        # Ollama 확인
        if not self.ollama.is_available():
            print("❌ Ollama가 실행되지 않음")
            print("   실행: ollama serve")
            return

        print(f"✅ 모델: {MODEL_NAME}")
        print(f"📚 학습 내용: {len(self.memory.learnings)}개")

        # 마이크 청취 시작
        if enable_mic:
            self.mic_listener = MicrophoneListener(self.on_voice_input)
            self.mic_listener.start()

        print("\n명령어:")
        print("  /quit - 종료")
        print("  /learn <내용> - 학습")
        print("  /memory - 메모리 상태")
        print("  /tools - 도구 목록")
        print("  /mic on/off - 마이크 토글")
        print("=" * 50 + "\n")

        self.running = True

        while self.running:
            try:
                # 음성 입력 확인
                try:
                    voice_text = self.input_queue.get_nowait()
                    self.chat(voice_text)
                    continue
                except queue.Empty:
                    pass

                # 텍스트 입력
                user_input = input("👤 You: ").strip()

                if not user_input:
                    continue

                # 명령어 처리
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # 대화
                self.chat(user_input)

            except KeyboardInterrupt:
                print("\n👋 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류: {e}")

        # 정리
        if self.mic_listener:
            self.mic_listener.stop()
        self.memory.save()

    def _handle_command(self, cmd: str):
        """명령어 처리"""
        if cmd == "/quit":
            self.running = False

        elif cmd.startswith("/learn "):
            content = cmd[7:]
            self.memory.add_learning("user_taught", content, "direct")
            print(f"💡 학습됨: {content}")

        elif cmd == "/memory":
            print(f"\n📚 메모리 상태:")
            print(f"  단기 기억: {len(self.memory.short_term)}개")
            print(f"  학습 내용: {len(self.memory.learnings)}개")
            print(f"  장기 지식: {len(self.memory.long_term)}개")

        elif cmd == "/tools":
            print("\n🔧 사용 가능한 도구:")
            for name in self.tools.tools:
                print(f"  - {name}")

        elif cmd == "/mic on":
            if self.mic_listener:
                self.mic_listener.start()
                print("🎤 마이크 활성화")

        elif cmd == "/mic off":
            if self.mic_listener:
                self.mic_listener.stop()
                print("🔇 마이크 비활성화")

        else:
            print("❓ 알 수 없는 명령어")


def main():
    """메인"""
    # requests 확인
    try:
        import requests
    except ImportError:
        print("❌ requests 미설치")
        print("   설치: pip3 install requests")
        return

    ai = RealtimeAI()
    ai.run(enable_mic=True)


if __name__ == "__main__":
    main()
