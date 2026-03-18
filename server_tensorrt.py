"""
Qwen3-ASR 推理服务 —— 不依赖 vLLM
====================================================
提供与主服务完全兼容的 API：
  POST /transcribe            批量音频转录
  WS   /transcribe-streaming  WebSocket 流式转录（滑动推理）
  GET  /health                健康检查

支持三种推理后端（ASR_BACKEND）：
  llm            纯 transformers，FP16，~3.5GB 显存，无 vLLM 依赖 （默认）
  torch_compile  同上 + torch.compile(inductor)，首次慢，后续内核融合加速
  torch_compile_trt  同上 + torch.compile(backend="tensorrt")，需安装 torch-tensorrt
                     首次推理会触发 TRT JIT 编译（比 ONNX 慢但无需静态图）
  tensorrt       已构建的 .engine 文件（需先完成 ONNX 导出，目前模型不支持）

流式原理（分片接收 + 分段推理）：
  - 客户端（如 UE 引擎）以分片(Chunk)方式发送音频：
    每次二进制帧仅包含当前时刻新采集的一小段 PCM（例如 50ms），
    而非从会话开始的累积数据。
  - 服务端将收到的分片追加到当前缓冲区，当样本数 >= STREAM_MIN_SAMPLES 时，
    对**当前缓冲区**做一次推理，将结果作为 partial 返回，随后立即清空缓冲区。
  - 不按固定时长积累（无 STREAM_CHUNK_SEC 等待），无服务端缓冲区上限。
  - 收到 stop 消息后对当前缓冲区剩余音频做最终推理 → 发 final。

启动示例：
  python server_tensorrt.py
  ASR_BACKEND=torch_compile python server_tensorrt.py
  ASR_BACKEND=torch_compile_trt python server_tensorrt.py
  ASR_BACKEND=tensorrt ASR_TENSORRT_ENGINE=qwen3_asr_1.7b.engine python server_tensorrt.py

UE HTTP /transcribe-ue 调用示例：
  # 启动服务（例如本地 8001 端口）
  LISTEN_PORT=8001 python server_tensorrt.py

  # 使用原始音频二进制作为请求体（Content-Type: application/octet-stream）
  curl -X POST \\
    -H "Content-Type: application/octet-stream" \\
    --data-binary @files/reference.wav \\
    http://127.0.0.1:8001/transcribe-ue

环境变量一览：
  ASR_BACKEND            llm | torch_compile | torch_compile_trt | tensorrt  (default: llm)
  ASR_MODEL_NAME         HuggingFace 模型 ID      (default: Qwen/Qwen3-ASR-1.7B)
  TORCH_DTYPE            float16 | bfloat16       (default: float16)
  ASR_TENSORRT_ENGINE    .engine 文件路径          (default: qwen3_asr_1.7b.engine)
  LISTEN_HOST                                     (default: 0.0.0.0)
  LISTEN_PORT                                     (default: 8001)
  MAX_CONCURRENT_DECODE  音频解码并发上限          (default: 4)
  MAX_CONCURRENT_INFER   GPU 推理并发上限          (default: 1)   #并发
  THREADPOOL_WORKERS     线程池大小                (default: cpu_count * 4)
  STREAM_MIN_SAMPLES     流式触发推理的最小样本数  (default: 1600, 即 100ms@16kHz)
  PARTIAL_INTERVAL_MS    partial 消息最小间隔(ms)  (default: 300)
  STREAM_EXPECT_SR       流式期望采样率            (default: 16000)
"""

# =============================================================================
# 标准库
# =============================================================================
import os
import io
import json
import asyncio
import logging
import subprocess
import time
from typing import Optional, List, Tuple, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# 第三方库
# =============================================================================
import uvicorn
import numpy as np
import soundfile as sf
import torch
import psutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =============================================================================
# 日志
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server_tensorrt")

# =============================================================================
# 工具函数
# =============================================================================

def get_env_bool(key: str, default: str = "true") -> bool:
    return os.getenv(key, default).lower() in ("true", "1", "yes", "on")


def map_language(lang_code: Optional[str]) -> Optional[str]:
    """ISO 语言代码 -> 模型语言名称"""
    if lang_code is None:
        return None

    # mapping = {
    #     "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    #     "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    #     "ru": "Russian", "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish",
    #     "sv": "Swedish", "id": "Indonesian", "vi": "Vietnamese",
    #     "hi": "Hindi", "ar": "Arabic",
    # }
    mapping = {"en": "English", "zh": "Chinese"}


    return mapping.get(lang_code.lower(), lang_code)


def read_audio_file(file_bytes: bytes) -> Tuple[np.ndarray, int]:   #是输出格式，不是输入限制，soundfile 内部会自动把任何格式（int16、int24、float32、float64）转成 float32 读出来。
    """同步解码音频：soundfile 优先，回退 ffmpeg"""
    try:
        with io.BytesIO(file_bytes) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr
    except Exception:
        proc = subprocess.Popen(
            ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-ac", "1", "-ar", "16000", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        out, err = proc.communicate(input=file_bytes)
        if proc.returncode != 0:
            raise ValueError(f"FFmpeg failed: {err.decode(errors='ignore')}")
        with io.BytesIO(out) as f:
            wav, sr = sf.read(f, dtype="float32", always_2d=False)
            return wav, sr

# =============================================================================
# 配置
# =============================================================================
ASR_BACKEND = os.getenv("ASR_BACKEND", "llm").lower()  # llm | torch_compile | torch_compile_trt | tensorrt
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-1.7B")
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float16")  # float16 | bfloat16
ASR_TENSORRT_ENGINE = os.getenv("ASR_TENSORRT_ENGINE", "qwen3_asr_1.7b.engine")

LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "8001"))

MAX_CONCURRENT_DECODE = int(os.getenv("MAX_CONCURRENT_DECODE", "4")) # 音频解码并发数，通常为4
MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "1"))  # GPU 推理并发数，通常为1
THREADPOOL_WORKERS = int(os.getenv("THREADPOOL_WORKERS", str((os.cpu_count() or 4) * 4)))

STREAM_MIN_SAMPLES = int(os.getenv("STREAM_MIN_SAMPLES", "16000"))   # 100ms @16kHz，达到即触发推理
STREAM_SILENCE_RMS = float(os.getenv("STREAM_SILENCE_RMS", "0"))    # 静音 chunk RMS 阈值，低于则丢弃；0=不过滤
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", "0"))  # partial 消息最小间隔(ms)300  不节流 为0
STREAM_EXPECT_SR = int(os.getenv("STREAM_EXPECT_SR", "16000"))

SAMPLE_RATE = 16000

# =============================================================================
# 全局状态
# =============================================================================
model: Any = None              # ASR 模型（PyTorch 或 TensorRT）
model_status = "starting"
model_ready_event = asyncio.Event()

decode_sem: asyncio.Semaphore
infer_sem: asyncio.Semaphore

# =============================================================================
# ASR 后端基础：结果对象
# =============================================================================

class TranscribeResult:
    """与 server.py 兼容的结果对象"""
    __slots__ = ("text", "language")
    def __init__(self, text: str, language: str = ""):
        self.text = text
        self.language = language


# =============================================================================
# ASR 后端 1：Qwen3ASRModel（transformers）+ 可选 torch.compile
# =============================================================================

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class QwenASRBackend:
    """
    基于 qwen_asr.Qwen3ASRModel.from_pretrained() 的推理后端。
    可选用 torch.compile() 对 thinker 子模块进行 JIT 内核融合加速。

    compile_backend 取值：
      None          — 纯 eager 推理（最稳定，~3.5GB 显存）
      "inductor"    — torch.compile 默认后端，首次慢，之后自动内核融合
      "tensorrt"    — torch.compile + TensorRT JIT（需安装 torch-tensorrt）
                      首次推理触发 TRT 编译（约 1-3 分钟），后续极快
    """

    def __init__(
        self,
        model_id: str,
        compile_backend: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise ImportError("请安装 qwen-asr: pip install qwen-asr") from e

        logger.info(f"加载 Qwen3ASRModel: {model_id}  dtype={dtype}  compile={compile_backend}")
        self._qwen = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cuda:0",
        )

        if compile_backend is not None:
            self._apply_compile(compile_backend, dtype)

        self.sample_rate = SAMPLE_RATE
        logger.info("QwenASRBackend 初始化完成")

    def _apply_compile(self, compile_backend: str, dtype: torch.dtype):
        """对 thinker 子模块应用 torch.compile"""
        if compile_backend == "tensorrt":
            try:
                import torch_tensorrt  # noqa: F401
            except ImportError as e:
                torch_ver = torch.__version__
                cu_tag = "cu" + torch_ver.split("+cu")[-1] if "+cu" in torch_ver else "cu128"
                raise ImportError(
                    f"使用 torch_compile_trt 后端需安装与当前 PyTorch ({torch_ver}) 匹配的 torch-tensorrt。\n\n"
                    "推荐安装方式（nightly，与当前 CUDA 版本匹配）：\n"
                    f"  pip install --pre torch-tensorrt "
                    f"--index-url https://download.pytorch.org/whl/nightly/{cu_tag}\n\n"
                    "若需稳定版，需将 PyTorch 升级到 2.10.x（cu130）：\n"
                    "  pip install torch==2.10.0 torchvision torchaudio "
                    "--index-url https://download.pytorch.org/whl/cu130\n"
                    "  pip install torch-tensorrt\n\n"
                    "或改用 torch_compile 后端（inductor，无需额外安装，已内置于 PyTorch）：\n"
                    "  ASR_BACKEND=torch_compile python server_tensorrt.py"
                ) from e
            options = {"enabled_precisions": {dtype}}
            logger.info("torch.compile(backend='tensorrt') 已应用到 thinker")
        else:
            options = None
            logger.info(f"torch.compile(backend='{compile_backend}') 已应用到 thinker")

        thinker = self._qwen.model.thinker
        self._qwen.model.thinker = torch.compile(
            thinker,
            backend=compile_backend,
            **({"options": options} if options else {}),
        )
        logger.info("  首次推理时将触发 JIT 编译，耗时较长（属正常现象）")

    def transcribe(
        self,
        audio: List[Tuple[np.ndarray, int]],
        language: Optional[List[Optional[str]]] = None,
        return_time_stamps: bool = False,
    ) -> List[TranscribeResult]:
        results = self._qwen.transcribe(
            audio=audio,
            language=language,
            return_time_stamps=return_time_stamps,
        )
        return [TranscribeResult(text=r.text, language=r.language) for r in results]


def _resample_audio(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    try:
        from scipy.signal import resample
        n = int(len(wav) * target_sr / orig_sr)
        return resample(wav, n).astype(np.float32)
    except ImportError:
        n = int(len(wav) * target_sr / orig_sr)
        return wav[:n] if len(wav) >= n else np.pad(wav, (0, n - len(wav)))


class TensorRTASRBackend:
    """
    TensorRT 引擎后端
    使用已构建的 .engine 文件，通过 pycuda 分配内存执行推理。
    """

    def __init__(self, engine_path: str, model_id: str):
        import tensorrt as trt_lib
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda_drv
        from transformers import AutoProcessor

        self._trt = trt_lib
        self._cuda = cuda_drv

        logger.info(f"加载 TensorRT 引擎: {engine_path}")
        trt_logger = trt_lib.Logger(trt_lib.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt_lib.Runtime(trt_logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._stream = cuda_drv.Stream()

        self._idx_input_f = self._engine.get_binding_index("input_features")
        self._idx_input_m = self._engine.get_binding_index("attention_mask")
        self._idx_logits = self._engine.get_binding_index("logits")

        logger.info(f"加载 Processor: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._tokenizer = getattr(self.processor, "tokenizer", None)
        if self._tokenizer is None or not hasattr(self._tokenizer, "decode"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.sample_rate = SAMPLE_RATE
        logger.info("TensorRTASRBackend 初始化完成")

    def _run_engine(self, input_features: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        cuda_drv = self._cuda

        self._context.set_binding_shape(self._idx_input_f, input_features.shape)
        self._context.set_binding_shape(self._idx_input_m, attention_mask.shape)

        out_shape = tuple(self._context.get_binding_shape(self._idx_logits))
        out_size = int(np.prod(out_shape))
        logits_nbytes = out_size * 4  # float32

        d_input_f = cuda_drv.mem_alloc(input_features.nbytes)
        d_input_m = cuda_drv.mem_alloc(attention_mask.nbytes)
        d_logits = cuda_drv.mem_alloc(logits_nbytes)

        cuda_drv.memcpy_htod_async(d_input_f, input_features, self._stream)
        cuda_drv.memcpy_htod_async(d_input_m, attention_mask, self._stream)

        n = self._engine.num_bindings
        bindings = [0] * n
        bindings[self._idx_input_f] = int(d_input_f)
        bindings[self._idx_input_m] = int(d_input_m)
        bindings[self._idx_logits] = int(d_logits)

        self._context.execute_async_v2(bindings, self._stream.handle)
        self._stream.synchronize()

        logits = np.empty(out_size, dtype=np.float32)
        cuda_drv.memcpy_dtoh_async(logits, d_logits, self._stream)
        self._stream.synchronize()

        return logits.reshape(out_shape)

    def _transcribe_one(self, wav: np.ndarray, sr: int, language: Optional[str] = None) -> str:
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        if sr != self.sample_rate:
            wav = _resample_audio(wav, sr, self.sample_rate)

        inputs = self.processor(
            audio=wav, sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
        )
        input_features = inputs.input_features.numpy().astype(np.float32)
        attention_mask = inputs.attention_mask.numpy().astype(np.int32)

        logits = self._run_engine(input_features, attention_mask)
        if logits.ndim == 3:
            logits = logits[0]

        ids = np.argmax(logits, axis=-1).flatten().tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True).strip()

    def transcribe(
        self,
        audio: List[Tuple[np.ndarray, int]],
        language: Optional[List[Optional[str]]] = None,
        return_time_stamps: bool = False,
    ) -> List[TranscribeResult]:
        if language is None:
            language = [None] * len(audio)
        results = []
        for i, (wav, sr) in enumerate(audio):
            lang = language[i] if i < len(language) else None
            text = self._transcribe_one(wav, sr, lang)
            results.append(TranscribeResult(text=text, language=lang or ""))
        return results


# =============================================================================
# 模型加载
# =============================================================================

async def load_model_background():
    """后台加载模型"""
    global model, model_status

    model_status = "loading_models"
    dtype = _DTYPE_MAP.get(TORCH_DTYPE, torch.float16)
    logger.info(f"后端: {ASR_BACKEND} | 模型: {ASR_MODEL_NAME} | dtype: {TORCH_DTYPE}")

    try:
        if ASR_BACKEND == "tensorrt":
            if not os.path.isfile(ASR_TENSORRT_ENGINE):
                raise FileNotFoundError(f"TensorRT engine not found: {ASR_TENSORRT_ENGINE}")
            model = await asyncio.to_thread(
                TensorRTASRBackend,
                engine_path=ASR_TENSORRT_ENGINE,
                model_id=ASR_MODEL_NAME,
            )
        elif ASR_BACKEND == "torch_compile_trt":
            model = await asyncio.to_thread(
                QwenASRBackend,
                model_id=ASR_MODEL_NAME,
                compile_backend="tensorrt",
                dtype=dtype,
            )
        elif ASR_BACKEND == "torch_compile":
            model = await asyncio.to_thread(
                QwenASRBackend,
                model_id=ASR_MODEL_NAME,
                compile_backend="inductor",
                dtype=dtype,
            )
        else:  # llm（默认）：纯 eager transformers，最省显存
            model = await asyncio.to_thread(
                QwenASRBackend,
                model_id=ASR_MODEL_NAME,
                compile_backend=None,
                dtype=dtype,
            )
    except Exception as e:
        logger.exception(f"模型加载失败: {e}")
        model_status = "error"
        model_ready_event.set()
        return

    # 预热
    logger.info("预热推理...")
    model_status = "warming_up"
    try:
        dummy_wav = np.zeros(SAMPLE_RATE, dtype=np.float32)
        await asyncio.to_thread(
            model.transcribe,
            audio=[(dummy_wav, SAMPLE_RATE)],
            language=["English"],
        )
        logger.info("预热完成")
    except Exception as e:
        logger.warning(f"预热失败 (non-critical): {e}")

    model_status = "ready"
    model_ready_event.set()
    logger.info("服务就绪，可以接受请求")


# =============================================================================
# 并发控制
# =============================================================================

async def to_thread_limited(sem: asyncio.Semaphore, fn, *args, **kwargs):
    async with sem:
        return await asyncio.to_thread(fn, *args, **kwargs)

# =============================================================================
# FastAPI 生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global decode_sem, infer_sem
    logger.info("启动 Qwen3-ASR TensorRT 服务...")

    decode_sem = asyncio.Semaphore(MAX_CONCURRENT_DECODE)
    infer_sem = asyncio.Semaphore(MAX_CONCURRENT_INFER)

    executor = ThreadPoolExecutor(max_workers=THREADPOOL_WORKERS)
    app.state.executor = executor
    asyncio.get_running_loop().set_default_executor(executor)

    task = asyncio.create_task(load_model_background())
    try:
        yield
    finally:
        task.cancel()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        executor.shutdown(wait=False, cancel_futures=True)
        logger.info("服务已关闭")


# =============================================================================
# FastAPI 应用
# =============================================================================

app = FastAPI(
    title="Qwen3-ASR TensorRT Server",
    description="不依赖 vLLM，纯 transformers/TensorRT 后端，支持流式和批量 ASR",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 端点: /health
# =============================================================================

@app.get("/health")
async def health():
    mem = psutil.virtual_memory()
    info = {
        "status": model_status,
        "backend": ASR_BACKEND,
        "dtype": TORCH_DTYPE,
        "model": ASR_MODEL_NAME,
        "limits": {
            "max_concurrent_decode": MAX_CONCURRENT_DECODE,
            "max_concurrent_infer": MAX_CONCURRENT_INFER,
            "threadpool_workers": THREADPOOL_WORKERS,
            "stream_min_samples": STREAM_MIN_SAMPLES,
            "stream_silence_rms": STREAM_SILENCE_RMS,
        },
        "memory": {
            "ram_total_mb": mem.total // (1024 * 1024),
            "ram_available_mb": mem.available // (1024 * 1024),
            "ram_percent": mem.percent,
        },
    }
    if torch.cuda.is_available():
        info["memory"]["gpu_allocated_mb"] = torch.cuda.memory_allocated() // (1024 * 1024)
        info["memory"]["gpu_reserved_mb"] = torch.cuda.memory_reserved() // (1024 * 1024)
    return info


# =============================================================================
# 端点: POST /transcribe（批量转录）
# =============================================================================

@app.post("/transcribe")
async def transcribe(
    files: List[UploadFile] = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g. en, de, fr, zh). None=auto."),
):
    await model_ready_event.wait()
    if model_status != "ready" or model is None:
        raise HTTPException(status_code=503, detail=f"Server not ready: {model_status}")

    full_lang = map_language(language)

    async def decode_one(f: UploadFile):
        content = await f.read()
        return await to_thread_limited(decode_sem, read_audio_file, content)

    try:
        audio_batch = await asyncio.gather(*(decode_one(f) for f in files))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    try:
        async with infer_sem:
            results = await asyncio.to_thread(
                model.transcribe,
                audio=audio_batch,
                language=[full_lang] * len(audio_batch),
            )
        return [{"text": r.text, "language": r.language} for r in results]
    except Exception as e:
        logger.exception(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 端点: WS /transcribe-streaming（流式转录）
# =============================================================================

@app.websocket("/transcribe-streaming")
async def websocket_streaming(
    ws: WebSocket,
    language: Optional[str] = Query(None),
):
    """
    WebSocket 流式转录

    只对当前缓冲区推理并返回结果，不积累、无服务端缓冲区上限：
    ─────────────────────────────────────────────
    客户端以分片方式推送音频（每次仅新采集的 PCM）。
    服务端将分片追加到当前缓冲区，当样本数 >= STREAM_MIN_SAMPLES 时，
    对当前缓冲区做一次推理，将结果作为 partial 返回并清空缓冲区；
    收到 stop 时对当前缓冲区做最终推理并发 final。
    """
    await ws.accept()

    # 等模型就绪
    await model_ready_event.wait()
    if model_status != "ready" or model is None:
        await ws.close(code=1011, reason=f"Server not ready: {model_status}")
        return

    full_lang = map_language(language)
    started = False

    # 发送就绪
    try:
        await ws.send_json({"type": "ready"})
    except Exception:
        return

    # ── 当前缓冲区：达到最小样本数即推理并清空，无积累、无上限 ──
    audio_buf: List[np.ndarray] = []
    audio_buf_n = 0
    last_partial_ts = 0.0

    async def do_inference_and_send(*, is_final: bool):
        """对当前缓冲区中的积累音频执行推理并发送结果"""
        nonlocal last_partial_ts

        if audio_buf_n == 0:
            if is_final:
                await ws.send_json({"type": "final", "text": "", "language": full_lang or ""})
            return

        # 合并当前段全部分片
        wav = np.concatenate(audio_buf) if len(audio_buf) > 1 else audio_buf[0]

        # GPU 推理
        async with infer_sem:
            results = await asyncio.to_thread(
                model.transcribe,
                audio=[(wav, SAMPLE_RATE)],
                language=[full_lang],
            )

        text = results[0].text if results else ""
        lang = results[0].language if results else (full_lang or "")

        if is_final:
            await ws.send_json({"type": "final", "text": text, "language": lang})
        else:
            now = time.monotonic()
            if (now - last_partial_ts) * 1000 >= PARTIAL_INTERVAL_MS:
                await ws.send_json({"type": "partial", "text": text, "language": lang})
                last_partial_ts = now

    # ── 消息循环 ──
    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break
            if msg["type"] != "websocket.receive":
                continue

            # ── JSON 控制消息 ──
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    data = None

                if isinstance(data, dict):
                    t = data.get("type")

                    if t == "start":
                        started = True
                        client_sr = int(data.get("sample_rate_hz", 0)) if data.get("sample_rate_hz") else None
                        fmt = data.get("format")

                        if client_sr != STREAM_EXPECT_SR or fmt not in (None, "pcm_s16le"):
                            await ws.send_json({
                                "type": "error",
                                "message": f"Only pcm_s16le @ {STREAM_EXPECT_SR}Hz supported",
                            })
                            await ws.close(code=1003)
                            return

                        if full_lang is not None:
                            await ws.send_json({"type": "info", "message": f"language={full_lang}"})
                        continue

                    if t == "stop":
                        await do_inference_and_send(is_final=True)
                        await ws.close(code=1000)
                        return

            # ── 二进制音频帧 ──
            if msg.get("bytes"):
                if not started:
                    await ws.send_json({
                        "type": "error",
                        "message": "Send {type:'start', format:'pcm_s16le', sample_rate_hz:16000} first",
                    })
                    await ws.close(code=1002)
                    return

                raw = msg["bytes"]
                pcm_i16 = np.frombuffer(raw, dtype=np.int16)
                if pcm_i16.size == 0:
                    continue

                # 客户端每次只发本次新增的分片（非累积），服务端追加到本地缓冲区
                chunk_f32 = pcm_i16.astype(np.float32) / 32768.0
                # 静音过滤：RMS 低于阈值则丢弃该 chunk，不进入缓冲区，避免空音频占满推理队列
                if STREAM_SILENCE_RMS > 0 and _chunk_rms(chunk_f32) < STREAM_SILENCE_RMS:
                    continue
                audio_buf.append(chunk_f32)
                audio_buf_n += chunk_f32.size

                # 当前缓冲区达到最小样本数即推理，只返回本段结果，然后清空
                if audio_buf_n >= STREAM_MIN_SAMPLES:
                    await do_inference_and_send(is_final=False)
                    audio_buf.clear()
                    audio_buf_n = 0

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception(f"WS Error: {e}")
        try:
            await ws.close(code=1011, reason="internal error")
        except Exception:
            pass

from fastapi import Request
@app.post("/transcribe-ue")
async def transcribe_ue(request: Request):
    """
    专为 UE Async HTTP Request 设计
    Body 直接是音频文件原始二进制，language 写死中文
    
    UE 端设置：
      URL:    http://192.168.1.169:8001/transcribe-ue
      Method: POST
      Header: Content-Type: application/octet-stream
      Body:   音频文件字节数组
    """
    await model_ready_event.wait()
    if model_status != "ready" or model is None:
        raise HTTPException(status_code=503, detail=f"Server not ready: {model_status}")

    file_bytes = await request.body()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty body")

    try:
        wav, sr = await to_thread_limited(decode_sem, read_audio_file, file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    try:
        async with infer_sem:
            results = await asyncio.to_thread(
                model.transcribe,
                audio=[(wav, sr)],
                language=["Chinese"],   # 写死中文
            )
        return {"text": results[0].text, "language": results[0].language}
    except Exception as e:
        logger.exception(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        log_level="warning",
    )
