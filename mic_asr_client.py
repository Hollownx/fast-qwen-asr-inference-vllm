"""
麦克风实时 WebSocket 流式转录测试客户端
=========================================
依赖安装：
  pip install websockets pyaudio numpy

用法：
  python mic_asr_client.py
  python mic_asr_client.py --host 192.168.1.100 --port 8001
  python mic_asr_client.py --language zh --chunk-ms 200 --threshold 16000

参数说明：
  --host          服务端地址           (default: 127.0.0.1)
  --port          服务端端口           (default: 8001)
  --language      语言代码 zh/en/...   (default: zh)
  --chunk-ms      每次发送的音频时长ms  (default: 50)
  --threshold     触发推理的样本数      (default: 16000, 即1秒@16kHz)
  --sample-rate   采样率               (default: 16000)
  --device-index  麦克风设备索引       (default: 系统默认)
  --list-devices  列出所有音频设备后退出
"""

import argparse
import asyncio
import json
import sys
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

# ── 依赖检查 ──────────────────────────────────────────────────────────────────
try:
    import pyaudio
except ImportError:
    print("[错误] 请安装 pyaudio：pip install pyaudio")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("[错误] 请安装 websockets：pip install websockets")
    sys.exit(1)

# =============================================================================
# 颜色输出工具（兼容 Windows）
# =============================================================================

def _supports_color():
    import os
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.name != "nt"

USE_COLOR = _supports_color()

def colored(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def c_green(t):  return colored(t, "32")
def c_yellow(t): return colored(t, "33")
def c_cyan(t):   return colored(t, "36")
def c_gray(t):   return colored(t, "90")
def c_red(t):    return colored(t, "31")
def c_bold(t):   return colored(t, "1")

# =============================================================================
# 麦克风采集线程
# =============================================================================

class MicrophoneCapture:
    """
    在独立线程中持续采集麦克风 PCM，
    将分片（int16 bytes）放入线程安全队列供异步消费。
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_ms: int = 100,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_frames = int(sample_rate * chunk_ms / 1000)  # 每次读取的帧数
        self.device_index = device_index

        self._pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._queue: deque = deque()          # 线程安全（GIL 保护的 deque）
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def list_devices():
        pa = pyaudio.PyAudio()
        print(c_bold("\n可用音频输入设备："))
        print(f"{'索引':>4}  {'名称':<40}  {'最大输入通道':>6}  {'默认采样率':>8}")
        print("─" * 68)
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                marker = c_green(" ◀ 默认") if i == pa.get_default_input_device_info()["index"] else ""
                print(f"{i:>4}  {info['name']:<40}  {int(info['maxInputChannels']):>6}  "
                      f"{int(info['defaultSampleRate']):>8}{marker}")
        pa.terminate()
        print()

    def start(self):
        self._running = True
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_frames,
        )
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(c_green(f"[麦克风] 开始采集  采样率={self.sample_rate}Hz  "
                      f"每帧={self.chunk_frames}samples"))

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()

    def _capture_loop(self):
        while self._running:
            try:
                data = self._stream.read(self.chunk_frames, exception_on_overflow=False)
                self._queue.append(data)
            except Exception as e:
                if self._running:
                    print(c_red(f"[麦克风] 采集错误: {e}"))
                break

    def get_chunk(self) -> Optional[bytes]:
        """非阻塞取一个分片，无数据返回 None"""
        try:
            return self._queue.popleft()
        except IndexError:
            return None

    def drain(self) -> list:
        """取出当前队列中所有待发分片"""
        chunks = []
        while True:
            try:
                chunks.append(self._queue.popleft())
            except IndexError:
                break
        return chunks

# =============================================================================
# 转录结果显示
# =============================================================================

class TranscriptDisplay:
    """管理终端上的转录结果显示（partial 原位刷新 + final 换行确认）"""

    def __init__(self):
        self._last_partial = ""
        self._finals: list[str] = []
        self._start_time = time.time()

    def show_partial(self, text: str):
        if text == self._last_partial:
            return
        self._last_partial = text
        # 用 \r 回到行首覆盖上一条 partial
        print(f"\r{c_yellow('◌')} {c_yellow(text):<60}", end="", flush=True)

    def show_final(self, text: str):
        self._last_partial = ""
        elapsed = time.time() - self._start_time
        ts = time.strftime("%H:%M:%S")
        # 清除当前行再打印 final
        print(f"\r{' ' * 65}\r", end="")
        if text:
            self._finals.append(text)
            print(f"{c_gray(ts)}  {c_green('●')} {c_bold(text)}")
        else:
            print(c_gray(f"[{ts}] (空结果)"))

    def show_info(self, msg: str):
        print(f"\r{' ' * 65}\r{c_cyan('[信息]')} {msg}")

    def show_error(self, msg: str):
        print(f"\r{' ' * 65}\r{c_red('[错误]')} {msg}")

    def summary(self):
        print("\n" + "═" * 60)
        print(c_bold("转录汇总："))
        for i, t in enumerate(self._finals, 1):
            print(f"  {c_gray(str(i)+'.')} {t}")
        if not self._finals:
            print(c_gray("  （无有效转录结果）"))
        print("═" * 60)

# =============================================================================
# WebSocket 客户端主逻辑
# =============================================================================

async def run_streaming_client(
    host: str,
    port: int,
    language: str,
    chunk_ms: int,
    sample_rate: int,
    device_index: Optional[int],
):
    url = f"ws://{host}:{port}/transcribe-streaming?language={language}"
    display = TranscriptDisplay()
    mic = MicrophoneCapture(
        sample_rate=sample_rate,
        chunk_ms=chunk_ms,
        device_index=device_index,
    )

    print(c_bold("\n=== Qwen3-ASR 流式转录客户端 ==="))
    print(f"  服务地址: {c_cyan(url)}")
    print(f"  语言:     {c_cyan(language)}")
    print(f"  分片时长: {c_cyan(str(chunk_ms) + 'ms')}")
    print(f"  采样率:   {c_cyan(str(sample_rate) + 'Hz')}")
    print(c_gray("\n按 Ctrl+C 停止录音并获取最终结果\n"))

    try:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB
        ) as ws:

            # ── 1. 等待服务端就绪 ────────────────────────────────────────────
            print(c_gray("[连接] 等待服务端就绪..."))
            ready_msg = await asyncio.wait_for(ws.recv(), timeout=30)
            ready = json.loads(ready_msg)
            if ready.get("type") != "ready":
                display.show_error(f"意外消息: {ready}")
                return
            print(c_green("[连接] 服务端就绪 ✓"))

            # ── 2. 发送 start 握手 ───────────────────────────────────────────
            await ws.send(json.dumps({
                "type": "start",
                "format": "pcm_s16le",
                "sample_rate_hz": sample_rate,
            }))

            # ── 3. 启动麦克风 ────────────────────────────────────────────────
            mic.start()
            print(c_bold("[录音中] 请说话..."))

            # ── 4. 并发：发送音频 + 接收结果 ─────────────────────────────────
            stop_event = asyncio.Event()

            async def sender():
                """持续从麦克风队列取数据发送给服务端"""
                while not stop_event.is_set():
                    chunks = mic.drain()
                    if chunks:
                        for chunk in chunks:
                            await ws.send(chunk)
                    else:
                        await asyncio.sleep(0.01)   # 无数据时让出控制权

            async def receiver():
                """持续接收服务端返回的转录结果"""
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    t = msg.get("type")
                    text = msg.get("text", "")
                    lang = msg.get("language", "")

                    if t == "partial":
                        display.show_partial(text)

                    elif t == "final":
                        display.show_final(text)
                        # final 后连接会关闭，退出循环
                        break

                    elif t == "info":
                        display.show_info(msg.get("message", ""))

                    elif t == "error":
                        display.show_error(msg.get("message", ""))
                        stop_event.set()
                        break

            # 启动发送协程（后台）
            send_task = asyncio.create_task(sender())

            # 主循环：接收消息，直到用户按 Ctrl+C
            recv_task = asyncio.create_task(receiver())

            try:
                await recv_task
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                # ── 5. 用户中断 → 发 stop → 等最终结果 ─────────────────────
                stop_event.set()
                send_task.cancel()

                print(f"\r{' ' * 65}\r")
                print(c_yellow("[停止] 正在发送剩余音频并获取最终结果..."))

                # 把队列里还没发的音频全部推送
                remaining = mic.drain()
                for chunk in remaining:
                    try:
                        await ws.send(chunk)
                    except Exception:
                        pass

                # 发送 stop 指令
                try:
                    await ws.send(json.dumps({"type": "stop"}))
                except Exception:
                    pass

                # 等待 final 消息（最多 10 秒）
                try:
                    async with asyncio.timeout(10):
                        async for raw in ws:
                            msg = json.loads(raw)
                            if msg.get("type") == "final":
                                display.show_final(msg.get("text", ""))
                                break
                except (asyncio.TimeoutError, Exception):
                    pass

    except ConnectionRefusedError:
        print(c_red(f"\n[错误] 无法连接到服务端 {host}:{port}"))
        print(c_gray("  请确认服务已启动：python server_tensorrt.py"))
        return
    except websockets.exceptions.InvalidURI:
        print(c_red(f"\n[错误] 无效的 WebSocket 地址: {url}"))
        return
    except Exception as e:
        print(c_red(f"\n[错误] 连接异常: {e}"))
        return
    finally:
        mic.stop()

    display.summary()

# =============================================================================
# 主入口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR WebSocket 流式转录麦克风测试客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host",         default="127.0.0.1", help="服务端地址 (default: 127.0.0.1)")
    parser.add_argument("--port",         default=8001, type=int, help="服务端端口 (default: 8001)")
    parser.add_argument("--language",     default="zh",  help="语言代码 (default: zh)")
    parser.add_argument("--chunk-ms",     default=50, type=int, help="每次发送的音频时长ms (default: 100)")
    parser.add_argument("--sample-rate",  default=16000, type=int, help="采样率 (default: 16000)")
    parser.add_argument("--device-index", default=None, type=int, help="麦克风设备索引 (default: 系统默认)")
    parser.add_argument("--list-devices", action="store_true", help="列出所有音频输入设备后退出")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        MicrophoneCapture.list_devices()
        return

    try:
        asyncio.run(
            run_streaming_client(
                host=args.host,
                port=args.port,
                language=args.language,
                chunk_ms=args.chunk_ms,
                sample_rate=args.sample_rate,
                device_index=args.device_index,
            )
        )
    except KeyboardInterrupt:
        print(c_gray("\n[退出]"))


if __name__ == "__main__":
    main()


#python mic_asr_client.py --host 192.168.1.169 --port 8001 --device-index 21
#python mic_asr_client.py --list-devices