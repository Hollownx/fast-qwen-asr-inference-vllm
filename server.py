# 标准库导入
"""
Qwen3-ASR 推理服务
====================================================
提供与主服务完全兼容的 API：
  POST /transcribe            批量音频转录
  WS   /transcribe-streaming  WebSocket 
  GET  /health                健康检查

vllm累加转录
"""
import os
import json
import io
import asyncio
import logging
import subprocess
from typing import Optional, List, Tuple
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import time

# 第三方库导入
import uvicorn
import numpy as np
import soundfile as sf
import torch
import psutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 导入 Qwen-ASR 组件
try:
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
except ImportError:
    print("Warning: qwen_asr not found.")
    Qwen3ASRModel = None
    Qwen3ForcedAligner = None

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# 配置参数
# -----------------------------
def get_env_bool(key: str, default: str = "true") -> bool:
    """从环境变量读取布尔值"""
    return os.getenv(key, default).lower() in ("true", "1", "yes", "on")

# 并发控制配置
MAX_CONCURRENT_DECODE = int(os.getenv("MAX_CONCURRENT_DECODE", "2"))  # 最大并发音频解码数
MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "1"))  # GPU推理并发数，通常为1
THREADPOOL_WORKERS = int(os.getenv("THREADPOOL_WORKERS", str((os.cpu_count() or 4) * 5)))  # 线程池工作线程数

# 流式传输缓冲/节流配置
STREAM_MIN_SAMPLES = int(os.getenv("STREAM_MIN_SAMPLES", "1600"))  # 流式处理最小样本数（16kHz下约100ms）
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", "120"))  # 部分结果发送间隔（毫秒）
STREAM_EXPECT_SR = int(os.getenv("STREAM_EXPECT_SR", "16000"))  # 流式传输期望采样率（Hz）

# -----------------------------
# 应用状态
# -----------------------------
models = {}  # 存储加载的模型（ASR和Aligner）
model_status = "starting"  # 模型状态：starting/loading_models/warming_up/ready/error
model_ready_event = asyncio.Event()  # 模型就绪事件，用于同步等待模型加载完成

# 信号量：控制并发数量
decode_sem = asyncio.Semaphore(MAX_CONCURRENT_DECODE)  # 音频解码并发控制
infer_sem = asyncio.Semaphore(MAX_CONCURRENT_INFER)  # GPU推理并发控制

# -----------------------------
# 辅助函数
# -----------------------------
async def to_thread_limited(sem: asyncio.Semaphore, fn, *args, **kwargs):
    """在信号量控制下将同步函数转换为异步执行"""
    async with sem:
        return await asyncio.to_thread(fn, *args, **kwargs)

def map_language(lang_code: Optional[str]) -> Optional[str]:
    """将ISO语言代码映射为Qwen模型所需的完整语言名称"""
    if lang_code is None:
        return None
    mapping = {
        "en": "English", "de": "German", "fr": "French", "es": "Spanish",
        "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
        "ru": "Russian", "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish",
        "sv": "Swedish", "id": "Indonesian", "vi": "Vietnamese",
        "hi": "Hindi", "ar": "Arabic",
    }
    return mapping.get(lang_code.lower(), lang_code)

def read_audio_file(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    同步解码音频文件
    必须通过 asyncio.to_thread 或线程池调用
    优先使用 soundfile，失败时回退到 ffmpeg（用于 mp3/m4a 等格式）
    
    返回: (音频数据数组, 采样率)
    """
    try:
        # 尝试使用 soundfile 直接读取
        with io.BytesIO(file_bytes) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr
    except Exception:
        # 使用 ffmpeg 解码（适用于 mp3、m4a 等格式）
        process = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = process.communicate(input=file_bytes)
        if process.returncode != 0:
            raise ValueError(f"FFmpeg decoding failed: {err.decode(errors='ignore')}")
        with io.BytesIO(out) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr

# -----------------------------
# 模型加载
# -----------------------------
async def load_models_background():
    """后台异步加载ASR和Aligner模型"""
    global model_status
    logger.info("Background task: Loading models...")
    model_status = "loading_models"

    async def _load_asr():
        """加载ASR（自动语音识别）模型"""
        global model_status
        if not get_env_bool("ENABLE_ASR_MODEL", "true"):
            logger.info("ASR Model disabled via ENABLE_ASR_MODEL.")
            return
        if Qwen3ASRModel is None:
            raise RuntimeError("qwen_asr not installed (Qwen3ASRModel missing).")

        model_name = os.getenv("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
        backend_type = os.getenv("ASR_BACKEND_TYPE", "vllm").lower()  # "vllm" 或 "transformers"
        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))

        logger.info(f"Loading ASR Model: {model_name} | backend={backend_type}")
        try:
            if backend_type == "transformers":
                # Transformers 后端：仅占模型权重显存（~3.5GB），无 KV-cache 预分配
                # 不支持流式转录，/transcribe-streaming 将返回 503
                torch_dtype = os.getenv("TORCH_DTYPE", "float16")
                dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
                models["asr"] = await asyncio.to_thread(
                    Qwen3ASRModel.from_pretrained,
                    model_name,
                    max_new_tokens=max_new_tokens,
                    dtype=dtype_map.get(torch_dtype, torch.float16),
                    device_map="cuda:0",
                )
                models["asr_backend_type"] = "transformers"
            else:
                # vLLM 后端：支持流式转录，但会预分配 KV-cache
                # 降显存关键参数：
                #   GPU_MEMORY_UTILIZATION=0.3  降低 KV-cache 池（默认 0.4）
                #   VLLM_MAX_MODEL_LEN=512      ASR 输出很短，无需 1024
                #   ENFORCE_EAGER=true          关闭 CUDA graph，省 1-2GB
                #   MAX_NUM_SEQS=4              限制并发序列数，减少 KV-cache 分配
                gpu_mem = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.3"))
                max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "512"))
                enforce_eager = get_env_bool("ENFORCE_EAGER", "true")
                max_num_seqs = int(os.getenv("MAX_NUM_SEQS", "4"))
                logger.info(
                    f"vLLM params: gpu_mem={gpu_mem} | max_model_len={max_model_len} "
                    f"| enforce_eager={enforce_eager} | max_num_seqs={max_num_seqs} "
                    f"| max_new_tokens={max_new_tokens}"
                )
                models["asr"] = await asyncio.to_thread(
                    Qwen3ASRModel.LLM,
                    model=model_name,
                    gpu_memory_utilization=gpu_mem,
                    max_new_tokens=max_new_tokens,
                    max_model_len=max_model_len,
                    enforce_eager=enforce_eager,
                    max_num_seqs=max_num_seqs,
                )
                models["asr_backend_type"] = "vllm"
            logger.info("ASR Model loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load ASR model: {e}")
            model_status = "error"
            raise

    async def _load_aligner():
        """加载强制对齐（时间戳对齐）模型"""
        global model_status
        if not get_env_bool("ENABLE_ALIGNER_MODEL", "true"):
            logger.info("Aligner Model disabled via ENABLE_ALIGNER_MODEL.")
            return
        if Qwen3ForcedAligner is None:
            raise RuntimeError("qwen_asr not installed (Qwen3ForcedAligner missing).")

        aligner_name = os.getenv("ALIGNER_MODEL_NAME", "Qwen/Qwen3-ForcedAligner-0.6B")
        logger.info(f"Loading Aligner Model: {aligner_name}...")

        try:
            models["aligner"] = await asyncio.to_thread(
                Qwen3ForcedAligner.from_pretrained,
                aligner_name,
                dtype=torch.bfloat16,  # 使用bfloat16精度
                device_map="cuda:0",  # 使用第一个GPU
            )
            logger.info("Aligner Model loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load Aligner model: {e}")
            model_status = "error"
            raise

    try:
        # 并行加载ASR和Aligner模型
        await asyncio.gather(_load_asr(), _load_aligner())
    except Exception:
        # 加载失败时，model_status已由加载函数设置为"error"
        model_ready_event.set()  # 设置事件，避免端点挂起
        return

    # 模型预热（尽力而为，失败不影响服务）
    if "asr" in models:
        logger.info("Warming up ASR model (best-effort)...")
        model_status = "warming_up"
        try:
            # 创建虚拟音频数据用于预热
            dummy_wav = np.zeros(16000, dtype=np.float32)  # 1秒静音（16kHz）
            dummy_sr = 16000

            # 预热批量转录
            async with infer_sem:
                await asyncio.to_thread(
                    models["asr"].transcribe,
                    audio=[(dummy_wav, dummy_sr)],
                    language=["English"],
                    return_time_stamps=False,
                )

            # 初始化流式状态
            async with infer_sem:
                state = await asyncio.to_thread(
                    models["asr"].init_streaming_state,
                    unfixed_chunk_num=2,
                    unfixed_token_num=5,
                    chunk_size_sec=2.0,
                )

            # 预热流式转录（使用不同大小的chunk）
            warmup_chunks = [320, 640, 1024, 3200] + [3200] * 25
            for n in warmup_chunks:
                async with infer_sem:
                    await asyncio.to_thread(models["asr"].streaming_transcribe, dummy_wav[:n], state)

            # 完成流式转录
            async with infer_sem:
                await asyncio.to_thread(models["asr"].finish_streaming_transcribe, state)

            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")

    model_status = "ready"
    model_ready_event.set()  # 通知所有等待的端点模型已就绪
    logger.info("Server is ready to accept requests.")

# -----------------------------
# 应用生命周期管理
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI应用生命周期管理器：启动时初始化，关闭时清理资源"""
    logger.info("Starting up Qwen3-ASR Server...")

    # 创建更大的线程池，有助于处理并发场景：音频解码 + WebSocket缓冲 + 其他to_thread调用
    executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)
    app.state.executor = executor
    asyncio.get_running_loop().set_default_executor(executor)

    # 在后台任务中加载模型
    task = asyncio.create_task(load_models_background())
    try:
        yield  # 应用运行期间
    finally:
        # 关闭清理
        task.cancel()  # 取消模型加载任务
        models.clear()  # 清空模型字典
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空GPU缓存
        executor.shutdown(wait=False, cancel_futures=True)  # 关闭线程池
        logger.info("Shutdown complete.")

# -----------------------------
# FastAPI应用实例
# -----------------------------
app = FastAPI(lifespan=lifespan)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# -----------------------------
# API端点
# -----------------------------
@app.get("/health")
async def health():
    """健康检查端点，返回服务器状态和资源使用情况"""
    mem = psutil.virtual_memory()
    info = {
        "status": model_status,  # 模型状态
        "limits": {
            "max_concurrent_decode": MAX_CONCURRENT_DECODE,  # 最大并发解码数
            "max_concurrent_infer": MAX_CONCURRENT_INFER,  # 最大并发推理数
            "threadpool_workers": THREADPOOL_WORKERS,  # 线程池工作线程数
        },
        "memory": {
            "ram_total_mb": mem.total // (1024 * 1024),  # RAM总量（MB）
            "ram_available_mb": mem.available // (1024 * 1024),  # RAM可用量（MB）
            "ram_percent": mem.percent,  # RAM使用百分比
        },
    }
    # 如果CUDA可用，添加GPU内存信息
    if torch.cuda.is_available():
        info["memory"]["gpu_allocated_mb"] = torch.cuda.memory_allocated() // (1024 * 1024)  # GPU已分配内存（MB）
        info["memory"]["gpu_reserved_mb"] = torch.cuda.memory_reserved() // (1024 * 1024)  # GPU保留内存（MB）
    return info

@app.post("/transcribe")
async def transcribe(
    files: List[UploadFile] = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g. en, de, fr). None for auto-detect."),
    forced_alignment: bool = Query(False, description="Enable forced alignment (timestamps)"),
):
    """
    批量音频转录端点
    支持多文件上传，可指定语言或自动检测，可选择是否进行强制对齐（时间戳）
    """
    # 等待模型加载完成
    await model_ready_event.wait()

    # 检查模型状态
    if model_status != "ready":
        raise HTTPException(status_code=503, detail=f"Server not ready: {model_status}")
    if "asr" not in models:
        raise HTTPException(status_code=503, detail="ASR model is not enabled or failed to load.")

    # 映射语言代码
    full_lang = map_language(language)

    async def decode_one(f: UploadFile):
        """解码单个音频文件"""
        content = await f.read()
        return await to_thread_limited(decode_sem, read_audio_file, content)

    # 并发解码音频文件（受信号量限制）
    try:
        audio_batch = await asyncio.gather(*(decode_one(f) for f in files))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # GPU推理（显式限制并发，因为GPU并发不是免费的）
    try:
        async with infer_sem:
            results = await asyncio.to_thread(
                models["asr"].transcribe,
                audio=audio_batch,
                language=[full_lang] * len(audio_batch),
                return_time_stamps=False,
            )

        response_list = []

        # 如果需要强制对齐（时间戳）
        if forced_alignment:
            if "aligner" not in models:
                raise HTTPException(status_code=503, detail="Aligner model is not enabled or failed to load.")

            texts = [r.text for r in results]

            # 执行强制对齐
            async with infer_sem:
                alignment_results = await asyncio.to_thread(
                    models["aligner"].align,
                    audio=audio_batch,
                    text=texts,
                    language=[full_lang] * len(audio_batch),
                )

            # 组合转录结果和时间戳
            for i, res in enumerate(results):
                response_list.append(
                    {"text": res.text, "language": res.language, "timestamps": alignment_results[i]}
                )
        else:
            # 仅返回转录文本和语言
            for res in results:
                response_list.append({"text": res.text, "language": res.language})

        return response_list

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/transcribe-streaming")
async def websocket_endpoint(
    ws: WebSocket,
    language: Optional[str] = Query(None),
    forced_alignment: bool = Query(False),  # 为API对称性保留，流式传输中尚未使用
):
    """
    WebSocket流式转录端点
    支持实时音频流输入，返回部分和最终转录结果
    """
    await ws.accept()

    # 等待模型加载完成
    await model_ready_event.wait()

    # 检查模型状态
    if model_status != "ready" or "asr" not in models:
        await ws.close(code=1011, reason=f"Server not ready: {model_status}")
        return

    # 流式转录仅支持 vLLM 后端
    if models.get("asr_backend_type") == "transformers":
        await ws.send_json({
            "type": "error",
            "message": "流式转录需要 vLLM 后端。请设置 ASR_BACKEND_TYPE=vllm（会占用更多显存），"
                       "或使用 /transcribe 接口进行批量转录。",
        })
        await ws.close(code=1003)
        return

    full_lang = map_language(language)
    client_sr = None  # 客户端采样率
    started = False  # 是否已开始接收音频

    # 初始化流式状态（在事件循环外执行，受GPU并发限制）
    try:
        async with infer_sem:
            state = await asyncio.to_thread(
                models["asr"].init_streaming_state,
                unfixed_chunk_num=2,  # 未固定chunk数量
                unfixed_token_num=5,  # 未固定token数量
                chunk_size_sec=2.0,  # chunk大小（秒）
            )
    except Exception as e:
        logger.exception(f"Failed to init streaming state: {e}")
        await ws.close(code=1011, reason="init_streaming_state failed")
        return

    # 发送就绪消息
    try:
        await ws.send_json({"type": "ready"})
    except Exception:
        return

    # 音频缓冲区
    buf_parts: List[np.ndarray] = []  # 缓冲的音频片段列表
    buf_n = 0  # 缓冲区总样本数
    last_partial_ts = 0.0  # 上次发送部分结果的时间戳

    async def flush_and_infer(send_partial: bool):
        """刷新缓冲区并执行推理，可选择是否发送部分结果"""
        nonlocal buf_parts, buf_n, last_partial_ts
        if buf_n <= 0:
            return
        # 合并所有缓冲片段
        chunk = np.concatenate(buf_parts, axis=0) if len(buf_parts) > 1 else buf_parts[0]

        # 执行流式转录
        async with infer_sem:
            await asyncio.to_thread(models["asr"].streaming_transcribe, chunk, state)

        # 清空缓冲区
        buf_parts.clear()
        buf_n = 0

        # 如果需要发送部分结果，且距离上次发送时间超过阈值
        if send_partial:
            now = time.monotonic()
            if (now - last_partial_ts) * 1000.0 >= PARTIAL_INTERVAL_MS:
                await ws.send_json({"type": "partial", "text": state.text, "language": state.language})
                last_partial_ts = now

    try:
        while True:
            msg = await ws.receive()

            # 处理断开连接
            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] != "websocket.receive":
                continue

            # 处理控制消息（JSON文本）
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    data = None

                if isinstance(data, dict):
                    t = data.get("type")

                    # 开始消息：初始化音频流
                    if t == "start":
                        started = True
                        client_sr = int(data.get("sample_rate_hz", 0)) if data.get("sample_rate_hz") else None
                        fmt = data.get("format")

                        # 验证格式：仅支持pcm_s16le @ 16kHz
                        if client_sr != STREAM_EXPECT_SR or fmt not in (None, "pcm_s16le"):
                            await ws.send_json(
                                {"type": "error", "message": f"Only pcm_s16le @ {STREAM_EXPECT_SR}Hz supported"}
                            )
                            await ws.close(code=1003)
                            return

                        # 可选：确认语言选择
                        if full_lang is not None:
                            await ws.send_json({"type": "info", "message": f"language={full_lang}"})
                        continue

                    # 停止消息：完成转录并返回最终结果
                    if t == "stop":
                        # 刷新剩余缓冲区，完成转录，发送最终结果
                        await flush_and_infer(send_partial=False)
                        async with infer_sem:
                            await asyncio.to_thread(models["asr"].finish_streaming_transcribe, state)

                        await ws.send_json({"type": "final", "text": state.text, "language": state.language})
                        await ws.close(code=1000)
                        return

            # 处理音频帧（二进制数据）
            if msg.get("bytes"):
                if not started:
                    # 要求显式发送start消息以验证格式
                    await ws.send_json({"type": "error", "message": "Send {type:'start', format:'pcm_s16le', sample_rate_hz:16000} first"})
                    await ws.close(code=1002)
                    return

                chunk_bytes = msg["bytes"]
                # 将int16单声道小端序转换为float32 [-1, 1]
                audio_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
                if audio_int16.size == 0:
                    continue

                # 归一化到[-1, 1]范围
                audio_f32 = audio_int16.astype(np.float32) / 32768.0
                buf_parts.append(audio_f32)
                buf_n += audio_f32.size

                # 当缓冲区达到最小样本数时，执行推理
                if buf_n >= STREAM_MIN_SAMPLES:
                    await flush_and_infer(send_partial=True)

    except WebSocketDisconnect:
        pass  # 客户端正常断开
    except Exception as e:
        logger.exception(f"WS Error: {e}")
        try:
            await ws.close(code=1011, reason="internal error")
        except Exception:
            pass

if __name__ == "__main__":
    # 注意：对于GPU模型，保持workers=1，除非您有意为每个worker复制模型
    uvicorn.run(app, host="0.0.0.0", port=8000)


## 24GB 卡上进一步压缩到最低
#make up GPU_MEM_UTIL=0.2 ENFORCE_EAGER=true MAX_NUM_SEQS=2
