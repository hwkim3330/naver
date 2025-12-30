#!/usr/bin/env python3
"""
HyperCLOVAX 하이브리드 AI
- 기본: Ollama (qwen2.5:3b) - 빠름
- 선택: HyperCLOVAX CPU - 느리지만 정확
- 마이크 청취
- 도구 실행
- 자가 학습
"""

import os
import sys
import json
import time
import queue
import threading
import subprocess
import tempfile
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")

# 설정
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434"
MEMORY_FILE = "/mnt/data/HyperCLOVAX-AGI/ai_memory.json"


@dataclass
class Memory:
    """AI 메모리"""
    conversations: List[Dict] = field(default_factory=list)
    learnings: List[Dict] = field(default_factory=list)
    tools_used: List[Dict] = field(default_factory=list)

    def save(self):
        data = {
            "conversations": self.conversations[-50:],
            "learnings": self.learnings[-200:],
            "tools_used": self.tools_used[-100:],
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
                self.tools_used = data.get("tools_used", [])
                print(f"📚 메모리 로드: {len(self.learnings)}개 학습")
            except:
                pass

    def add_conversation(self, role: str, content: str):
        self.conversations.append({
            "role": role,
            "content": content,
            "time": datetime.now().isoformat()
        })

    def learn(self, category: str, content: str):
        self.learnings.append({
            "category": category,
            "content": content,
            "time": datetime.now().isoformat()
        })
        print(f"💡 학습됨: [{category}] {content[:50]}...")

    def log_tool(self, tool: str, args: str, result: str):
        self.tools_used.append({
            "tool": tool,
            "args": args,
            "result": result[:500],
            "time": datetime.now().isoformat()
        })


class Tools:
    """도구 모음"""

    @staticmethod
    def bash(cmd: str) -> str:
        """Bash 명령 실행"""
        try:
            r = subprocess.run(
                cmd, shell=True,
                capture_output=True, text=True,
                timeout=30
            )
            return (r.stdout + r.stderr)[:2000] or "[완료]"
        except subprocess.TimeoutExpired:
            return "[타임아웃]"
        except Exception as e:
            return f"[오류: {e}]"

    @staticmethod
    def python(code: str) -> str:
        """Python 실행"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                path = f.name
            r = subprocess.run(
                ['python3', path],
                capture_output=True, text=True,
                timeout=30
            )
            os.unlink(path)
            return (r.stdout + r.stderr)[:2000] or "[완료]"
        except Exception as e:
            return f"[오류: {e}]"

    @staticmethod
    def read_file(path: str) -> str:
        """파일 읽기"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()[:5000]
        except Exception as e:
            return f"[오류: {e}]"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        """파일 쓰기"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"[저장: {path}]"
        except Exception as e:
            return f"[오류: {e}]"

    @staticmethod
    def list_dir(path: str = ".") -> str:
        """디렉토리 목록"""
        try:
            files = os.listdir(path)
            return "\n".join(sorted(files)[:50])
        except Exception as e:
            return f"[오류: {e}]"

    @staticmethod
    def system_info() -> str:
        """시스템 정보"""
        import platform
        info = [
            f"OS: {platform.system()} {platform.release()}",
            f"Python: {platform.python_version()}",
        ]
        try:
            gpu = subprocess.run(
                "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader",
                shell=True, capture_output=True, text=True, timeout=5
            )
            if gpu.stdout:
                info.append(f"GPU: {gpu.stdout.strip()}")
        except:
            pass
        return "\n".join(info)

    @staticmethod
    def time_now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def calc(expr: str) -> str:
        """계산"""
        try:
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expr):
                return "[허용되지 않는 문자]"
            return str(eval(expr))
        except Exception as e:
            return f"[오류: {e}]"

    @staticmethod
    def web_search(query: str) -> str:
        """웹 검색 (구현 필요)"""
        return f"[검색: {query}] - API 키 필요"


class MicListener:
    """마이크 청취"""

    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.running = False
        self.thread = None

        try:
            import speech_recognition as sr
            self.sr = sr
            self.recognizer = sr.Recognizer()
            self.available = True
        except ImportError:
            self.available = False
            print("⚠️ speech_recognition 미설치")

    def start(self):
        if not self.available:
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
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)

                try:
                    text = self.recognizer.recognize_google(audio, language="ko-KR")
                    if text:
                        print(f"\n🎤 [{text}]")
                        self.callback(text)
                except self.sr.UnknownValueError:
                    pass
                except self.sr.RequestError:
                    pass
            except:
                time.sleep(0.5)


class OllamaEngine:
    """Ollama 추론 엔진"""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self.url = OLLAMA_URL

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=3)
            return r.status_code == 200
        except:
            return False

    def generate(self, prompt: str, system: str = None, stream: bool = True) -> str:
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        if system:
            data["system"] = system

        try:
            if stream:
                r = requests.post(
                    f"{self.url}/api/generate",
                    json=data, stream=True, timeout=60
                )
                response = ""
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            text = chunk["response"]
                            print(text, end="", flush=True)
                            response += text
                        if chunk.get("done"):
                            break
                print()
                return response
            else:
                r = requests.post(f"{self.url}/api/generate", json=data, timeout=60)
                return r.json().get("response", "")
        except Exception as e:
            return f"[Ollama 오류: {e}]"


class HybridAI:
    """하이브리드 AI 시스템"""

    def __init__(self):
        self.memory = Memory()
        self.memory.load()

        self.tools = Tools()
        self.engine = OllamaEngine()

        self.input_queue = queue.Queue()
        self.mic = None
        self.running = False

        # 시스템 프롬프트
        self.system_prompt = """당신은 HyperCLOVAX 스타일의 한국어 AI 어시스턴트입니다.

## 도구 사용
다음 형식으로 도구를 사용할 수 있습니다:
{{TOOL:tool_name:argument}}

사용 가능한 도구:
- bash: 쉘 명령 - {{TOOL:bash:ls -la}}
- python: 파이썬 실행 - {{TOOL:python:print(1+1)}}
- read_file: 파일 읽기 - {{TOOL:read_file:/path/to/file}}
- write_file: 파일 쓰기 - {{TOOL:write_file:/path:content}}
- list_dir: 디렉토리 목록 - {{TOOL:list_dir:/home}}
- system_info: 시스템 정보 - {{TOOL:system_info:}}
- time: 현재 시간 - {{TOOL:time:}}
- calc: 계산 - {{TOOL:calc:3*5+2}}

## 학습
새로운 것을 배우면 다음 형식으로 저장:
{{LEARN:category:content}}

예시:
- {{LEARN:user_preference:사용자는 간결한 답변을 선호함}}
- {{LEARN:code_pattern:이 프로젝트는 Python 3.12 사용}}

## 지침
- 한국어로 자연스럽게 대화
- 필요하면 도구 사용
- 중요한 정보는 학습으로 저장
- 간결하고 정확하게 답변
"""

    def process_response(self, response: str) -> str:
        """응답 처리 - 도구 실행, 학습 저장"""
        import re

        # 도구 실행: {{TOOL:name:arg}}
        def exec_tool(match):
            tool_name = match.group(1)
            arg = match.group(2) if match.group(2) else ""

            tool_map = {
                "bash": lambda a: self.tools.bash(a),
                "python": lambda a: self.tools.python(a),
                "read_file": lambda a: self.tools.read_file(a),
                "list_dir": lambda a: self.tools.list_dir(a if a else "."),
                "system_info": lambda a: self.tools.system_info(),
                "time": lambda a: self.tools.time_now(),
                "calc": lambda a: self.tools.calc(a),
            }

            if tool_name in tool_map:
                result = tool_map[tool_name](arg)
                self.memory.log_tool(tool_name, arg, result)
                return f"\n📌 [{tool_name}] 결과:\n```\n{result}\n```\n"
            return f"[알 수 없는 도구: {tool_name}]"

        response = re.sub(r'\{\{TOOL:(\w+):([^}]*)\}\}', exec_tool, response)

        # 학습: {{LEARN:category:content}}
        def save_learn(match):
            category = match.group(1)
            content = match.group(2)
            self.memory.learn(category, content)
            return ""

        response = re.sub(r'\{\{LEARN:(\w+):([^}]*)\}\}', save_learn, response)

        return response

    def build_context(self) -> str:
        """컨텍스트 구성"""
        parts = []

        # 최근 학습
        if self.memory.learnings:
            learns = self.memory.learnings[-10:]
            learn_text = "\n".join([
                f"- [{l['category']}] {l['content']}"
                for l in learns
            ])
            parts.append(f"[학습된 지식]\n{learn_text}")

        # 최근 대화
        if self.memory.conversations:
            convs = self.memory.conversations[-6:]
            conv_text = "\n".join([
                f"{'User' if c['role']=='user' else 'AI'}: {c['content']}"
                for c in convs
            ])
            parts.append(f"[대화 기록]\n{conv_text}")

        return "\n\n".join(parts)

    def chat(self, user_input: str) -> str:
        """대화"""
        self.memory.add_conversation("user", user_input)

        context = self.build_context()
        prompt = f"""{context}

User: {user_input}
AI:"""

        print("\n🤖 ", end="", flush=True)
        response = self.engine.generate(prompt, system=self.system_prompt)

        # 도구/학습 처리
        processed = self.process_response(response)
        if processed != response:
            print(processed)

        self.memory.add_conversation("assistant", response)
        self.memory.save()

        return response

    def on_voice(self, text: str):
        """음성 입력"""
        self.input_queue.put(text)

    def run(self, enable_mic: bool = True):
        """실행"""
        print("\n" + "=" * 50)
        print("  🤖 HyperCLOVAX 하이브리드 AI")
        print("=" * 50)

        # Ollama 확인
        if not self.engine.is_available():
            print("❌ Ollama 미실행. 시작: ollama serve")
            return

        print(f"✅ 모델: {OLLAMA_MODEL}")
        print(f"📚 학습: {len(self.memory.learnings)}개")

        # 마이크
        if enable_mic:
            self.mic = MicListener(self.on_voice)
            self.mic.start()

        print("\n명령어:")
        print("  /quit      - 종료")
        print("  /learn     - 학습 내용 보기")
        print("  /tools     - 최근 도구 사용")
        print("  /clear     - 대화 초기화")
        print("  /mic on|off - 마이크 토글")
        print("  /teach <내용> - 직접 학습")
        print("=" * 50 + "\n")

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

                # 명령어
                if user_input.startswith("/"):
                    self._handle_cmd(user_input)
                    continue

                # 대화
                self.chat(user_input)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ {e}")

        print("👋 종료")
        if self.mic:
            self.mic.stop()
        self.memory.save()

    def _handle_cmd(self, cmd: str):
        """명령어 처리"""
        if cmd == "/quit":
            self.running = False

        elif cmd == "/learn":
            print("\n📚 최근 학습:")
            for l in self.memory.learnings[-10:]:
                print(f"  [{l['category']}] {l['content'][:60]}")

        elif cmd == "/tools":
            print("\n🔧 최근 도구 사용:")
            for t in self.memory.tools_used[-5:]:
                print(f"  {t['tool']}: {t['args'][:40]}")

        elif cmd == "/clear":
            self.memory.conversations = []
            print("🗑️ 대화 초기화")

        elif cmd == "/mic on":
            if self.mic:
                self.mic.start()

        elif cmd == "/mic off":
            if self.mic:
                self.mic.stop()
                print("🔇 마이크 중지")

        elif cmd.startswith("/teach "):
            content = cmd[7:]
            self.memory.learn("user_taught", content)

        else:
            print("❓ 알 수 없는 명령어")


def main():
    ai = HybridAI()
    ai.run(enable_mic=True)


if __name__ == "__main__":
    main()
