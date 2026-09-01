"""
Qwen3-ASR 推理服务 —— 不依赖 vLLM
====================================================
提供与主服务完全兼容的 API：
  POST /transcribe            批量音频转录
                              可选 query: vocal_isolation_strength / rms_threshold
                              （与 /transcribe-ue 同名参数同语义；传参才启用预处理，不传则不做任何预处理）
                              可选 query: hotwords（逗号分隔热词，拼音模糊匹配后处理纠错；
                              不传时使用环境变量 ASR_HOTWORDS，传空串则本请求关闭热词；
                              纯后处理，所有后端通用，无 prompt 注入）
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


http://127.0.0.1:8001/transcribe-ue?vocal_isolation_strength=1.25&rms_threshold=0.4

/transcribe-ue 同款预处理也可用于 /transcribe（仅传参时启用）：
  curl -X POST -F "files=@files/reference.wav" \
    "http://127.0.0.1:8001/transcribe?language=zh&vocal_isolation_strength=1.25&rms_threshold=0.4"

环境变量一览：
  ASR_BACKEND            llm | torch_compile | torch_compile_trt | tensorrt  (default: llm)
  ASR_MODEL_NAME         HuggingFace 模型 ID/本地路径 (default: Qwen/Qwen3-ASR-0.6B)
  TORCH_DTYPE            float16 | bfloat16       (default: float16)
  ASR_TENSORRT_ENGINE    .engine 文件路径          (default: qwen3_asr_1.7b.engine)
  HF_LOCAL_ONLY          HuggingFace 仅本地离线加载 (default: true)
  HF_MODEL_CACHE_DIR     HuggingFace 本地缓存根目录 (default: hf_models)
  LISTEN_HOST                                     (default: 0.0.0.0)
  LISTEN_PORT                                     (default: 8001)
  MAX_CONCURRENT_DECODE  音频解码并发上限          (default: 4)
  MAX_CONCURRENT_INFER   GPU 推理并发上限          (default: 1)   #并发
  THREADPOOL_WORKERS     线程池大小                (default: cpu_count * 4)
  STREAM_MIN_SAMPLES     流式触发推理的最小样本数  (default: 1600, 即 100ms@16kHz)
  PARTIAL_INTERVAL_MS    partial 消息最小间隔(ms)  (default: 300)
  STREAM_EXPECT_SR       流式期望采样率            (default: 16000)
  EXPORT_INPUT_AUDIO     是否导出输入音频          (default: true)
  EXPORT_INPUT_AUDIO_DIR 导出目录                  (default: exported_audio)
  UE_ENABLE_VOCAL_ISOLATION    /transcribe-ue 人声分离预处理开关   (default: false)
  UE_EXPORT_PREPROCESSED_AUDIO 是否导出预处理后音频               (default: 同 EXPORT_INPUT_AUDIO)
  UE_VOCAL_ISOLATION_STRENGTH  人声分离强度                       (default: 1.25)
  UE_ENABLE_THRESHOLD_FILTER   /transcribe-ue 阈值过滤开关         (default: true)
  UE_THRESHOLD_FILTER_RMS      /transcribe-ue RMS 阈值             (default: 0.4)
  UE_THRESHOLD_FILTER_FRAME_MS 阈值过滤帧长(ms)                   (default: 20)
  UE_THRESHOLD_FILTER_PAD_MS   阈值过滤保留边界(ms)               (default: 120)
  TRANSCRIBE_FILTER_FILLERS    /transcribe 填充词后过滤开关        (default: true)
  ASR_HOTWORDS                 /transcribe 默认热词列表(逗号分隔, 拼音模糊纠错) (default: 昆明盲哑学校等 6 词)
  ASR_HOTWORD_FIXES            精确替换表 错词=对词(逗号分隔, 优先于拼音纠错) (default: 空)
"""

# =============================================================================
# 标准库
# =============================================================================
import os
import io
import json
import re
import asyncio
import logging
import subprocess
import time
import wave
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Tuple, Any
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from uuid import uuid4

# =============================================================================
# 第三方库
# =============================================================================
import uvicorn
import numpy as np
import soundfile as sf
import torch
import psutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Query, HTTPException, Request
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

    # 当前服务只显式映射最常用的几种语言，其他值透传给下游模型。
    mapping = {"en": "English", "zh": "Chinese"}
    return mapping.get(lang_code.lower(), lang_code)


def read_audio_file(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """同步解码音频，统一输出为 `float32` waveform + sample rate。"""
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
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "Qwen/Qwen3-ASR-0.6B")
TORCH_DTYPE = os.getenv("TORCH_DTYPE", "float16")  # float16 | bfloat16
ASR_TENSORRT_ENGINE = os.getenv("ASR_TENSORRT_ENGINE", "qwen3_asr_1.7b.engine")
HF_LOCAL_ONLY = get_env_bool("HF_LOCAL_ONLY", "true")
HF_MODEL_CACHE_DIR = Path(os.getenv("HF_MODEL_CACHE_DIR", "hf_models"))

# 服务监听配置
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", "8001"))

# 输入音频导出与 UE 预处理配置  <= 0：关闭该步骤
EXPORT_INPUT_AUDIO = get_env_bool("EXPORT_INPUT_AUDIO", "false")
EXPORT_INPUT_AUDIO_DIR = Path(os.getenv("EXPORT_INPUT_AUDIO_DIR", "exported_audio"))
UE_EXPORT_PREPROCESSED_AUDIO = get_env_bool(
    "UE_EXPORT_PREPROCESSED_AUDIO",
    "true" if EXPORT_INPUT_AUDIO else "false",
)
# UE 人声增强配置  越大：背景压得越狠，但也更容易把弱语音、尾音、混响一起削掉。
UE_ENABLE_VOCAL_ISOLATION = get_env_bool("UE_ENABLE_VOCAL_ISOLATION", "true")
UE_VOCAL_ISOLATION_STRENGTH = float(os.getenv("UE_VOCAL_ISOLATION_STRENGTH", "1.5")) #0.6～3.0

# UE RMS 阈值过滤配置 越大越削人声
UE_ENABLE_THRESHOLD_FILTER = get_env_bool("UE_ENABLE_THRESHOLD_FILTER", "true")
UE_THRESHOLD_FILTER_RMS = float(os.getenv("UE_THRESHOLD_FILTER_RMS", "0.35"))
UE_THRESHOLD_FILTER_FRAME_MS = int(os.getenv("UE_THRESHOLD_FILTER_FRAME_MS", "50"))  #检测窗口时长
UE_THRESHOLD_FILTER_PAD_MS = int(os.getenv("UE_THRESHOLD_FILTER_PAD_MS", "500")) #音频填充/缓冲时长

# /transcribe 填充词后过滤：ASR 结果仅由填充词（嗯啊呃等）和句号「。」组成时归为空文本
# （针对极短/静音片段的「嗯」类幻觉做兜底），可通过环境变量关闭
TRANSCRIBE_FILTER_FILLERS = get_env_bool("TRANSCRIBE_FILTER_FILLERS", "true")

# /transcribe 热词后处理（拼音模糊匹配纠错，替代 LLM context 注入）：
# 客户端不传 hotwords 时使用该服务级默认热词列表（逗号分隔）；为空则不做热词纠错
ASR_HOTWORDS = os.getenv("ASR_HOTWORDS", "昆明盲哑学校, 盲人，聋哑人，哑人，聋哑，盲哑，云南")

# 精确替换表（优先于拼音模糊纠错执行）：格式 "错词=对词"，逗号分隔
# 例: "盲哑人=盲哑人士,哑校=聋哑学校"；为空则不做精确替换
ASR_HOTWORD_FIXES = os.getenv("ASR_HOTWORD_FIXES", "昆明满雅学校=昆明盲哑学校,昆明玛雅学校=昆明盲哑学校")
 
# 并发与线程池配置
MAX_CONCURRENT_DECODE = int(os.getenv("MAX_CONCURRENT_DECODE", "4"))  # 音频解码并发数
MAX_CONCURRENT_INFER = int(os.getenv("MAX_CONCURRENT_INFER", "20"))  # GPU 推理并发数
THREADPOOL_WORKERS = int(os.getenv("THREADPOOL_WORKERS", str((os.cpu_count() or 4) * 4)))

# 流式接口配置
STREAM_MIN_SAMPLES = int(os.getenv("STREAM_MIN_SAMPLES", "16000"))  # 达到该样本数即触发推理
STREAM_SILENCE_RMS = float(os.getenv("STREAM_SILENCE_RMS", "0"))  # 低于该 RMS 的 chunk 会被丢弃
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", "0"))  # partial 最小间隔；0 表示不节流
STREAM_EXPECT_SR = int(os.getenv("STREAM_EXPECT_SR", "16000"))

SAMPLE_RATE = 16000

#下载过后
if HF_LOCAL_ONLY:
    # 强制让 transformers / huggingface_hub 只看本地目录和缓存，不做联网探测。
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


# =============================================================================
# HuggingFace 本地离线加载辅助
# =============================================================================


def _resolve_hf_snapshot_dir(cache_root: Path, model_id: str) -> Optional[Path]:
    repo_parts = [part for part in model_id.strip("/").split("/") if part]
    if len(repo_parts) < 2:
        return None

    repo_cache_dir = cache_root / f"models--{'--'.join(repo_parts)}"
    snapshots_dir = repo_cache_dir / "snapshots"
    refs_main = repo_cache_dir / "refs" / "main"

    if refs_main.is_file():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = snapshots_dir / revision
        if snapshot_dir.is_dir():
            return snapshot_dir.resolve()

    if snapshots_dir.is_dir():
        snapshots = sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return snapshots[0].resolve()

    return None


def resolve_hf_model_source(model_id: str) -> Tuple[str, dict]:
    """
    优先将 repo id 解析到工作区本地目录或 hf 缓存 snapshot，尽量不让 HF 代码碰网络。
    找不到明确本地目录时，再退回 repo id + local_files_only/cache_dir。
    """
    expanded = Path(model_id).expanduser()
    if expanded.is_dir():
        return str(expanded.resolve()), {}

    local_dir = Path(model_id.split("/")[-1])
    if local_dir.is_dir():
        return str(local_dir.resolve()), {}

    for cache_root in (HF_MODEL_CACHE_DIR, HF_MODEL_CACHE_DIR / "hub"):
        snapshot_dir = _resolve_hf_snapshot_dir(cache_root, model_id)
        if snapshot_dir is not None:
            return str(snapshot_dir), {}

    hf_kwargs = {}
    if HF_MODEL_CACHE_DIR.exists():
        hf_kwargs["cache_dir"] = str(HF_MODEL_CACHE_DIR.resolve())
    if HF_LOCAL_ONLY:
        hf_kwargs["local_files_only"] = True
    return model_id, hf_kwargs

# =============================================================================
# 音频导出辅助
# =============================================================================


def _sanitize_name(name: Optional[str], default: str = "audio") -> str:
    raw = Path(name).stem if name else default
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return cleaned or default


def _build_audio_dump_path(source: str, original_name: Optional[str] = None, suffix: str = ".wav") -> Path:
    safe_source = _sanitize_name(source, default="source")
    safe_name = _sanitize_name(original_name, default="audio")
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    return EXPORT_INPUT_AUDIO_DIR / f"{ts}_{ms:03d}_{safe_source}_{safe_name}_{uuid4().hex[:8]}{suffix}"


def dump_audio_file(
    wav: np.ndarray,
    sr: int,
    source: str,
    original_name: Optional[str] = None,
    *,
    enabled: bool = True,
) -> Optional[Path]:
    """将输入音频导出为 PCM16 WAV，失败时只记录日志，不影响主流程。"""
    if not enabled:
        return None

    try:
        EXPORT_INPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path = _build_audio_dump_path(source=source, original_name=original_name, suffix=".wav")
        wav = np.asarray(wav)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        sf.write(str(path), wav, sr, subtype="PCM_16")
        logger.info("已导出输入音频: %s", path)
        return path
    except Exception as e:
        logger.warning("导出输入音频失败(source=%s, original_name=%s): %s", source, original_name, e)
        return None


class StreamingAudioDumper:
    """将流式 PCM16LE 音频边接收边写入 WAV，避免在内存中累积完整会话。"""

    def __init__(self, path: Path, sample_rate: int):
        self.path = path
        self.sample_rate = sample_rate
        self.frames = 0
        self._writer = wave.open(str(path), "wb")
        self._writer.setnchannels(1)
        self._writer.setsampwidth(2)
        self._writer.setframerate(sample_rate)

    def write(self, raw_pcm_s16le: bytes):
        if not raw_pcm_s16le:
            return
        self._writer.writeframes(raw_pcm_s16le)
        self.frames += len(raw_pcm_s16le) // 2

    def close(self):
        writer = getattr(self, "_writer", None)
        if writer is None:
            return
        writer.close()
        self._writer = None
        logger.info(
            "已导出流式输入音频: %s (%d samples @ %dHz)",
            self.path,
            self.frames,
            self.sample_rate,
        )


def create_streaming_audio_dumper(source: str, sample_rate: int) -> Optional[StreamingAudioDumper]:
    if not EXPORT_INPUT_AUDIO:
        return None

    try:
        EXPORT_INPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path = _build_audio_dump_path(source=source, original_name="stream", suffix=".wav")
        return StreamingAudioDumper(path=path, sample_rate=sample_rate)
    except Exception as e:
        logger.warning("创建流式音频导出器失败(source=%s): %s", source, e)
        return None


# =============================================================================
# UE /transcribe-ue 预处理辅助
# =============================================================================


@dataclass(frozen=True)
class UEPreprocessConfig:
    vocal_isolation_enabled: bool
    vocal_isolation_strength: float
    threshold_filter_enabled: bool
    threshold_filter_rms: float
    threshold_filter_frame_ms: int
    threshold_filter_pad_ms: int


def resolve_ue_preprocess_config(
    vocal_isolation_strength: Optional[float] = None,
    rms_threshold: Optional[float] = None,
) -> UEPreprocessConfig:
    """
    解析一次 /transcribe-ue 请求的有效预处理配置。
    规则：
    - query 参数为 `None`：沿用服务默认值。
    - query 参数 `<= 0`：视为关闭该步骤。
    - query 参数 `> 0`：启用该步骤，并使用该值。
    """
    if vocal_isolation_strength is None:
        effective_vocal_strength = max(0.0, UE_VOCAL_ISOLATION_STRENGTH)
        vocal_enabled = UE_ENABLE_VOCAL_ISOLATION and effective_vocal_strength > 0
    else:
        effective_vocal_strength = max(0.0, vocal_isolation_strength)
        vocal_enabled = effective_vocal_strength > 0

    if rms_threshold is None:
        effective_rms_threshold = max(0.0, UE_THRESHOLD_FILTER_RMS)
        threshold_enabled = UE_ENABLE_THRESHOLD_FILTER and effective_rms_threshold > 0
    else:
        effective_rms_threshold = max(0.0, rms_threshold)
        threshold_enabled = effective_rms_threshold > 0

    return UEPreprocessConfig(
        vocal_isolation_enabled=vocal_enabled,
        vocal_isolation_strength=effective_vocal_strength,
        threshold_filter_enabled=threshold_enabled,
        threshold_filter_rms=effective_rms_threshold,
        threshold_filter_frame_ms=max(5, UE_THRESHOLD_FILTER_FRAME_MS),
        threshold_filter_pad_ms=max(0, UE_THRESHOLD_FILTER_PAD_MS),
    )


def resolve_transcribe_preprocess_config(
    vocal_isolation_strength: Optional[float],
    rms_threshold: Optional[float],
) -> Optional[UEPreprocessConfig]:
    """
    /transcribe 请求级预处理配置（方案A：按需开启）：
    - 未传（None）：该步骤关闭，不继承 UE_* 环境变量默认值。
    - 传 >0：启用该步骤并使用该值。
    - 传 <=0：显式关闭该步骤。
    返回 None 表示本请求完全不做预处理（与历史行为一致）。
    """
    if vocal_isolation_strength is None and rms_threshold is None:
        return None
    return resolve_ue_preprocess_config(
        vocal_isolation_strength=0.0 if vocal_isolation_strength is None else vocal_isolation_strength,
        rms_threshold=0.0 if rms_threshold is None else rms_threshold,
    )


def _chunk_rms(wav: np.ndarray) -> float:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return 0.0
    wav64 = wav.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(wav64 * wav64)))


_ASR_FILLER_CHARS = set("嗯啊呃哦噢喔唉诶欸呣嘛吧呢吗呀")


def is_filler_only_text(text: str) -> bool:
    """
    /transcribe 填充词后过滤判定：
    去掉首尾空白和句号「。」后，剩余字符全部属于填充词集合（如「嗯」「啊。」「嗯。。」）返回 True。
    """
    core = (text or "").strip().strip("。")
    if not core:
        return False
    return all(ch in _ASR_FILLER_CHARS for ch in core)


def _safe_audio(wav: np.ndarray) -> np.ndarray:
    wav = np.nan_to_num(np.asarray(wav, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(wav, -1.0, 1.0).astype(np.float32, copy=False)


def _mix_to_mono_for_speech_focus(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        return _safe_audio(wav)
    if wav.ndim == 2:
        if wav.shape[1] == 1:
            return _safe_audio(wav[:, 0])
        if wav.shape[1] >= 2:
            left = wav[:, 0]
            right = wav[:, 1]
            center = 0.5 * (left + right)
            if wav.shape[1] > 2:
                extra = np.mean(wav[:, 2:], axis=1, dtype=np.float32)
                center = 0.85 * center + 0.15 * extra
            return _safe_audio(center)
    return _safe_audio(wav.reshape(-1))


def _apply_sos_filter(sos: np.ndarray, wav: np.ndarray) -> np.ndarray:
    from scipy.signal import sosfilt, sosfiltfilt

    if wav.size < 32:
        return wav.astype(np.float32, copy=False)

    try:
        return sosfiltfilt(sos, wav).astype(np.float32)
    except ValueError:
        return sosfilt(sos, wav).astype(np.float32)


def _apply_ue_vocal_isolation(
    wav: np.ndarray,
    sr: int,
    reference_wav: np.ndarray,
    config: UEPreprocessConfig,
) -> np.ndarray:
    """
    轻量级“人声增强”而非严格源分离，目标是提升 ASR 前景语音占比。
    处理顺序：
    1. 带通滤波，先聚焦语音主频段。
    2. 基于短时频谱估计噪声底，做软掩蔽抑制背景。
    3. 用输入 RMS 做一次温和补偿，避免增强后音量掉太多。
    """
    try:
        from scipy.ndimage import uniform_filter
        from scipy.signal import butter, istft, stft
    except ImportError:
        logger.warning("UE 人声分离预处理跳过: scipy 不可用")
        return wav

    try:
        nyquist = sr * 0.5
        low_hz = max(80.0, min(120.0, nyquist * 0.45))
        high_hz = min(4800.0, nyquist - 120.0)
        if high_hz <= low_hz + 100.0:
            return wav

        strength = max(0.6, min(3.0, config.vocal_isolation_strength))
        peak = float(np.max(np.abs(wav)))
        norm = wav / peak if peak > 1.0 else wav.copy()

        # 第 1 步：先做语音频段带通，削掉明显无关的低频/高频能量。
        sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
        bandpassed = _apply_sos_filter(sos, norm)

        nperseg = int(2 ** np.round(np.log2(max(256, min(2048, sr * 0.032)))))
        nperseg = min(nperseg, bandpassed.size)
        if nperseg < 128:
            return _safe_audio(bandpassed)

        noverlap = min(nperseg - 1, nperseg * 3 // 4)
        _, _, zxx = stft(bandpassed, fs=sr, nperseg=nperseg, noverlap=noverlap)
        if zxx.size == 0:
            return _safe_audio(bandpassed)

        # 第 2 步：用低能量帧估计噪声底，再做平滑后的软掩蔽。
        mag = np.abs(zxx)
        frame_energy = np.mean(mag, axis=0)
        noise_frames = np.argsort(frame_energy)[:max(6, frame_energy.size // 8)]
        noise_profile = np.median(mag[:, noise_frames], axis=1, keepdims=True)
        threshold = noise_profile * strength

        mask = np.clip((mag - threshold) / (mag + 1e-6), 0.0, 1.0)
        mask = uniform_filter(mask, size=(3, 5), mode="nearest")

        freqs = np.linspace(0.0, nyquist, mag.shape[0], dtype=np.float32)
        speech_weight = np.where(
            (freqs >= low_hz) & (freqs <= high_hz),
            1.0,
            0.35,
        ).astype(np.float32)[:, None]
        mask = np.clip(mask * speech_weight, 0.02, 1.0)

        # 第 3 步：逆 STFT 回时域，并做一次温和的能量回补。
        _, enhanced = istft(zxx * mask, fs=sr, nperseg=nperseg, noverlap=noverlap)
        if enhanced.size < bandpassed.size:
            enhanced = np.pad(enhanced, (0, bandpassed.size - enhanced.size))
        enhanced = enhanced[:bandpassed.size]

        processed = 0.8 * enhanced.astype(np.float32) + 0.2 * bandpassed
        in_rms = _chunk_rms(reference_wav)
        out_rms = _chunk_rms(processed)
        if in_rms > 1e-6 and out_rms > 1e-6:
            processed = processed * min(3.0, in_rms / out_rms)

        processed = _safe_audio(processed)
        logger.info(
            "UE 人声分离预处理完成: sr=%d samples=%d rms_in=%.6f rms_out=%.6f strength=%.2f",
            sr,
            reference_wav.size,
            in_rms,
            _chunk_rms(processed),
            strength,
        )
        return processed
    except Exception as e:
        logger.warning("UE 人声分离预处理失败，回退到原始单声道: %s", e)
        return wav


def apply_ue_threshold_filter(wav: np.ndarray, sr: int, config: UEPreprocessConfig) -> np.ndarray:
    """
    基于短时 RMS 的门限过滤。
    低于阈值的短帧会被衰减到 0，同时通过前后扩张保留语音边界，避免吃字头字尾。
    """
    wav = _safe_audio(wav)
    if wav.size == 0 or not config.threshold_filter_enabled:
        return wav

    threshold = max(0.0, config.threshold_filter_rms)
    if threshold <= 0.0 or sr <= 0:
        return wav

    frame_ms = max(5, config.threshold_filter_frame_ms)
    pad_ms = max(0, config.threshold_filter_pad_ms)
    frame_len = max(64, int(sr * frame_ms / 1000))
    hop = max(32, frame_len // 2)

    if wav.size <= frame_len:
        rms = _chunk_rms(wav)
        if rms < threshold:
            # fail-open: 宁可不过滤，也绝不给 ASR 喂纯静音
            logger.warning(
                "UE 阈值过滤: 短音频整体 RMS %.6f 低于阈值 %.6f，跳过过滤",
                rms,
                threshold,
            )
        return wav

    frame_starts = list(range(0, wav.size, hop))
    frame_rms = np.empty(len(frame_starts), dtype=np.float32)
    for i, start in enumerate(frame_starts):
        end = min(start + frame_len, wav.size)
        frame_rms[i] = _chunk_rms(wav[start:end])

    keep = frame_rms >= threshold
    if np.any(keep):
        pad_frames = int(np.ceil((pad_ms / 1000.0) * sr / hop))
        if pad_frames > 0:
            kernel = np.ones(pad_frames * 2 + 1, dtype=np.int16)
            keep = np.convolve(keep.astype(np.int16), kernel, mode="same") > 0
    else:
        # fail-open: 阈值过高或音频过静时跳过过滤，绝不返回全零静音给 ASR
        logger.warning(
            "UE 阈值过滤后无保留帧，阈值过高或音频过静: samples=%d sr=%d threshold=%.6f max_frame_rms=%.6f，跳过过滤",
            wav.size,
            sr,
            threshold,
            float(frame_rms.max()),
        )
        return wav

    gate = np.zeros(wav.size, dtype=np.float32)
    weight = np.zeros(wav.size, dtype=np.float32)
    window = np.hanning(frame_len).astype(np.float32)
    if frame_len <= 2:
        window = np.ones(frame_len, dtype=np.float32)
    else:
        window = np.maximum(window, 1e-3)

    for keep_flag, start in zip(keep, frame_starts):
        end = min(start + frame_len, wav.size)
        win = window[: end - start]
        weight[start:end] += win
        if keep_flag:
            gate[start:end] += win

    mask = np.divide(gate, weight, out=np.zeros_like(gate), where=weight > 1e-6)
    filtered = wav * mask
    logger.info(
        "UE 阈值过滤完成: sr=%d samples=%d threshold=%.6f kept_ratio=%.3f",
        sr,
        wav.size,
        threshold,
        float(np.mean(keep.astype(np.float32))),
    )
    return _safe_audio(filtered)


def preprocess_ue_audio_for_asr(
    wav: np.ndarray,
    sr: int,
    config: Optional[UEPreprocessConfig] = None,
) -> np.ndarray:
    """
    /transcribe-ue 专用轻量预处理：
    1. 多声道中心提取/转单声道
    2. 可选的人声增强
    3. 可选的阈值过滤

    这不是 Demucs 级别的深度源分离，但对 ASR 前的人声增强更轻、更稳。
    """
    config = config or resolve_ue_preprocess_config()
    mono = _mix_to_mono_for_speech_focus(wav)
    if mono.size == 0:
        return mono
    processed = mono

    # 主步骤 1：如果开启了人声增强，优先提升语音和背景的分离度。
    if config.vocal_isolation_enabled and sr >= 4000 and mono.size >= max(512, sr // 20):
        processed = _apply_ue_vocal_isolation(processed, sr, reference_wav=mono, config=config)

    # 主步骤 2：再做门限过滤，压掉低能量噪声段。
    if config.threshold_filter_enabled:
        processed = apply_ue_threshold_filter(processed, sr, config)

    return _safe_audio(processed)

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


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _syllable_edit_distance(a: List[str], b: List[str]) -> int:
    """两个等长音节序列的编辑距离（小规模 DP）。"""
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


def split_hotword_terms(hotwords: Optional[str]) -> List[str]:
    """热词串解析：逗号/中文逗号/分号/换行分隔，去空项。"""
    if not hotwords or not hotwords.strip():
        return []
    for sep in ("，", ";", "；", "\n"):
        hotwords = hotwords.replace(sep, ",")
    return [t.strip() for t in hotwords.split(",") if t.strip()]


def split_hotword_fixes(fixes: Optional[str]) -> List[Tuple[str, str]]:
    """精确替换表解析：'错词=对词' 逗号分隔（兼容中文逗号/分号/换行）；无 '=' 的项忽略。"""
    if not fixes or not fixes.strip():
        return []
    for sep in ("，", ";", "；", "\n"):
        fixes = fixes.replace(sep, ",")
    pairs: List[Tuple[str, str]] = []
    for item in fixes.split(","):
        item = item.strip()
        if "=" not in item:
            continue
        wrong, right = item.split("=", 1)
        wrong, right = wrong.strip(), right.strip()
        if wrong and right:
            pairs.append((wrong, right))
    return pairs


@lru_cache(maxsize=512)
def _pinyin_of(text: str) -> Tuple[str, ...]:
    """无调拼音转换（带缓存；pypinyin 缺失时返回空元组）。"""
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return ()
    return tuple(lazy_pinyin(text))


def apply_hotword_correction(text: str, terms: List[str]) -> Tuple[str, List[Tuple[str, str, int]]]:
    """
    热词后处理纠错：
    - 纯中文热词：文本逐字转无调拼音（带缓存），音节级滑窗编辑距离命中则回写热词原文。
      音节数 <2 的热词跳过（防单字到处误命中）；命中要求字符连续、跨热词不重叠。
    - 纯中文热词：文本逐字转无调拼音（带缓存），音节级滑窗比对，仅「完全同音」命中才回写热词原文。
      音节数 <2 的热词跳过；命中要求字符连续、跨热词不重叠。
      词表保护：词表自带词及其 >=2 字连续子串不做替换（防近音热词互相吞噬）。
      中文热词按长度降序扫描（最长匹配优先），按词长分层容错：
      2~3 音节仅完全同音；>=4 音节容 1 个音节差（口音容错，长窗口误撞概率极低）。
      更重的固定口音错误用 ASR_HOTWORD_FIXES 精确替换表显式配置。
    - 非中文热词：大小写不敏感的子串替换。
    返回 (纠错后文本, [(识别片段, 热词, 距离), ...])；pypinyin 不可用时原样返回。
    """
    if not text or not terms:
        return text, []
    if _pinyin_of("测") == ():
        logger.warning("pypinyin 不可用，跳过热词拼音纠错")
        return text, []

    chars = list(text)
    replacements: List[Tuple[int, int, str]] = []  # (start, end_exclusive, term)
    applied: List[Tuple[str, str]] = []
    occupied: List[Tuple[int, int]] = []

    def spans_overlap(s: int, e: int) -> bool:
        return any(not (e <= os_ or s >= oe) for os_, oe in occupied)

    syls: List[str] = []
    syl_char_idx: List[int] = []
    for i, ch in enumerate(chars):
        if _is_cjk(ch):
            py = _pinyin_of(ch)
            if py:
                syls.append(py[0])
                syl_char_idx.append(i)

    # 词表保护集：词表自带词及其 >=2 字连续子串不做模糊替换，
    # 防止近音热词互相吞噬（如词表同时含 盲人/哑人 时，先遍历者吃掉后者）
    cjk_terms = [t for t in terms if t and all(_is_cjk(c) for c in t)]
    other_terms = [t for t in terms if t and not all(_is_cjk(c) for c in t)]
    vocab_exact = set(cjk_terms)
    protected: set = set()
    for t in cjk_terms:
        for i in range(len(t)):
            for j in range(i + 2, len(t) + 1):
                protected.add(t[i:j])

    def _scan_cjk_term(term: str, max_dist: int) -> None:
        t_syls = [py for c in term for py in _pinyin_of(c)]
        n = len(t_syls)
        if n < 2 or n > len(syls):
            return
        p = 0
        while p + n <= len(syls):
            if syl_char_idx[p + n - 1] - syl_char_idx[p] != n - 1:
                p += 1  # 窗口内字符不连续(夹标点等)，跳过
                continue
            cs, ce = syl_char_idx[p], syl_char_idx[p + n - 1] + 1
            if spans_overlap(cs, ce):
                p += 1
                continue
            window_text = "".join(chars[cs:ce])
            if window_text in protected:
                if window_text in vocab_exact:
                    logger.info("/transcribe 热词保护: %r 为词表自带词, 不做模糊替换", window_text)
                occupied.append((cs, ce))
                p += n
                continue
            d = _syllable_edit_distance(syls[p:p + n], t_syls)
            if d <= max_dist:
                original = "".join(chars[cs:ce])
                if original != term:
                    replacements.append((cs, ce, term))
                    applied.append((original, term, d))
                occupied.append((cs, ce))
                p += n
            else:
                p += 1

    # 单遍扫描，长度降序（最长匹配优先，避免短词碎片化长词窗口）。
    # 按词长分层容错：2~3 音节仅完全同音（短窗口模糊实测必跨词误伤，
    # 如「崖位」→「哑人」、「龙崖位」→「聋哑人」）；>=4 音节容 1 个音节差（口音容错，
    # 长窗口误撞概率极低，如「昆明满亚学校」→「昆明盲哑学校」）。
    # 更重的固定口音错误用 ASR_HOTWORD_FIXES 显式配置。
    for term in sorted(cjk_terms, key=lambda t: -len(t)):
        _scan_cjk_term(term, 0 if len(term) < 4 else 1)

    # 非中文热词：大小写不敏感的子串替换
    for term in other_terms:
        low = text.lower()
        t = term.lower()
        start = 0
        while True:
            idx = low.find(t, start)
            if idx < 0:
                break
            if not spans_overlap(idx, idx + len(term)):
                original = text[idx: idx + len(term)]
                if original != term:
                    replacements.append((idx, idx + len(term), term))
                    applied.append((original, term, 0))
                occupied.append((idx, idx + len(term)))
            start = idx + len(term)

    if not replacements:
        return text, []
    replacements.sort(key=lambda r: r[0])
    out: List[str] = []
    last = 0
    for s, e, term in replacements:
        out.append(text[last:s])
        out.append(term)
        last = e
    out.append(text[last:])
    return "".join(out), applied


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

        model_source, hf_kwargs = resolve_hf_model_source(model_id)
        logger.info(
            "加载 Qwen3ASRModel: requested=%s resolved=%s dtype=%s compile=%s local_only=%s",
            model_id,
            model_source,
            dtype,
            compile_backend,
            HF_LOCAL_ONLY,
        )
        self._qwen = Qwen3ASRModel.from_pretrained(
            model_source,
            dtype=dtype,
            device_map="cuda:0",
            **hf_kwargs,
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
        context: Optional[str] = None,
    ) -> List[TranscribeResult]:
        results = self._qwen.transcribe(
            audio=audio,
            language=language,
            return_time_stamps=return_time_stamps,
            context=context or "",
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

    _hotword_warned = False

    def __init__(self, engine_path: str, model_id: str):
        import tensorrt as trt_lib
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda_drv
        from transformers import AutoProcessor

        self._trt = trt_lib
        self._cuda = cuda_drv
        model_source, hf_kwargs = resolve_hf_model_source(model_id)

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

        logger.info(
            "加载 Processor/Tokenizer: requested=%s resolved=%s local_only=%s",
            model_id,
            model_source,
            HF_LOCAL_ONLY,
        )
        self.processor = AutoProcessor.from_pretrained(model_source, **hf_kwargs)
        self._tokenizer = getattr(self.processor, "tokenizer", None)
        if self._tokenizer is None or not hasattr(self._tokenizer, "decode"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_source, **hf_kwargs)

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
        context: Optional[str] = None,
    ) -> List[TranscribeResult]:
        if context and not TensorRTASRBackend._hotword_warned:
            logger.warning("TensorRT 后端不支持热词偏置(context)，已忽略（仅提示一次）")
            TensorRTASRBackend._hotword_warned = True
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
    if EXPORT_INPUT_AUDIO:
        logger.info("输入音频导出已开启: %s", EXPORT_INPUT_AUDIO_DIR.resolve())
    else:
        logger.info("输入音频导出已关闭")
    logger.info(
        "HF 离线本地加载: local_only=%s cache_dir=%s",
        HF_LOCAL_ONLY,
        HF_MODEL_CACHE_DIR.resolve(),
    )
    logger.info(
        "/transcribe-ue 预处理: vocal_isolation=%s strength=%.2f threshold_filter=%s threshold_rms=%.6f frame_ms=%d pad_ms=%d export_preprocessed=%s",
        UE_ENABLE_VOCAL_ISOLATION,
        UE_VOCAL_ISOLATION_STRENGTH,
        UE_ENABLE_THRESHOLD_FILTER,
        UE_THRESHOLD_FILTER_RMS,
        UE_THRESHOLD_FILTER_FRAME_MS,
        UE_THRESHOLD_FILTER_PAD_MS,
        UE_EXPORT_PREPROCESSED_AUDIO,
    )

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
        "hf_loading": {
            "local_only": HF_LOCAL_ONLY,
            "cache_dir": str(HF_MODEL_CACHE_DIR),
        },
        "limits": {
            "max_concurrent_decode": MAX_CONCURRENT_DECODE,
            "max_concurrent_infer": MAX_CONCURRENT_INFER,
            "threadpool_workers": THREADPOOL_WORKERS,
            "stream_min_samples": STREAM_MIN_SAMPLES,
            "stream_silence_rms": STREAM_SILENCE_RMS,
        },
        "input_audio_dump": {
            "enabled": EXPORT_INPUT_AUDIO,
            "dir": str(EXPORT_INPUT_AUDIO_DIR),
        },
        "ue_audio_preprocess": {
            "vocal_isolation": UE_ENABLE_VOCAL_ISOLATION,
            "export_preprocessed_audio": UE_EXPORT_PREPROCESSED_AUDIO,
            "strength": UE_VOCAL_ISOLATION_STRENGTH,
            "threshold_filter": {
                "enabled": UE_ENABLE_THRESHOLD_FILTER,
                "rms": UE_THRESHOLD_FILTER_RMS,
                "frame_ms": UE_THRESHOLD_FILTER_FRAME_MS,
                "pad_ms": UE_THRESHOLD_FILTER_PAD_MS,
            },
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
    vocal_isolation_strength: Optional[float] = Query(
        None,
        description="Request-level vocal isolation strength (0.6~3.0). <= 0 disables it for this request. None=off.",
    ),
    rms_threshold: Optional[float] = Query(
        None,
        description="Request-level RMS threshold. <= 0 disables threshold filtering for this request. None=off.",
    ),
    hotwords: Optional[str] = Query(
        None,
        description="Comma-separated hotwords overriding env ASR_HOTWORDS; applied as pinyin-fuzzy post-correction. Empty string disables for this request. None=use ASR_HOTWORDS.",
    ),
):
    await model_ready_event.wait()
    if model_status != "ready" or model is None:
        raise HTTPException(status_code=503, detail=f"Server not ready: {model_status}")

    full_lang = map_language(language)
    hotword_terms = split_hotword_terms(hotwords if hotwords is not None else ASR_HOTWORDS)
    hotword_fixes = split_hotword_fixes(ASR_HOTWORD_FIXES)

    # 仅当请求显式传入预处理参数时才启用（与 /transcribe-ue 同名参数同语义）
    preprocess_config = resolve_transcribe_preprocess_config(
        vocal_isolation_strength=vocal_isolation_strength,
        rms_threshold=rms_threshold,
    )
    if preprocess_config is not None:
        logger.info(
            "/transcribe 请求启用预处理: vocal_isolation_strength=%s rms_threshold=%s -> vocal_enabled=%s vocal_strength=%.3f threshold_enabled=%s threshold_rms=%.6f",
            vocal_isolation_strength,
            rms_threshold,
            preprocess_config.vocal_isolation_enabled,
            preprocess_config.vocal_isolation_strength,
            preprocess_config.threshold_filter_enabled,
            preprocess_config.threshold_filter_rms,
        )

    async def decode_one(idx: int, f: UploadFile):
        content = await f.read()
        wav, sr = await to_thread_limited(decode_sem, read_audio_file, content)
        if preprocess_config is not None:
            wav = await to_thread_limited(
                decode_sem, preprocess_ue_audio_for_asr, wav, sr, preprocess_config
            )
        await asyncio.to_thread(
            dump_audio_file,
            wav,
            sr,
            f"transcribe_{idx}",
            f.filename,
            enabled=EXPORT_INPUT_AUDIO,
        )
        return wav, sr

    try:
        audio_batch = await asyncio.gather(*(decode_one(idx, f) for idx, f in enumerate(files)))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    try:
        async with infer_sem:
            results = await asyncio.to_thread(
                model.transcribe,
                audio=audio_batch,
                language=[full_lang] * len(audio_batch),
            )
        out = []
        for r in results:
            text = r.text
            if TRANSCRIBE_FILTER_FILLERS and is_filler_only_text(text):
                if text.strip():
                    logger.info("/transcribe 填充词过滤: %r -> 空文本", text)
                text = ""
            if text:
                for wrong, right in hotword_fixes:
                    if wrong in text:
                        logger.info("/transcribe 热词精确替换: %r -> %r", wrong, right)
                        text = text.replace(wrong, right)
                if text and hotword_terms:
                    text, applied = apply_hotword_correction(text, hotword_terms)
                    for orig, term, dist in applied:
                        logger.info("/transcribe 热词替换: %r -> %r (dist=%d)", orig, term, dist)
            out.append({"text": text, "language": r.language})
        return out
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
    stream_audio_dumper: Optional[StreamingAudioDumper] = None

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

                        if stream_audio_dumper is not None:
                            stream_audio_dumper.close()
                        stream_audio_dumper = create_streaming_audio_dumper(
                            source="transcribe_streaming",
                            sample_rate=STREAM_EXPECT_SR,
                        )

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
                if stream_audio_dumper is not None:
                    stream_audio_dumper.write(raw)
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
    finally:
        if stream_audio_dumper is not None:
            try:
                stream_audio_dumper.close()
            except Exception as e:
                logger.warning("关闭流式音频导出器失败: %s", e)


@app.post("/transcribe-ue")
async def transcribe_ue(
    request: Request,
    vocal_isolation_strength: Optional[float] = Query(
        None,
        description="Request-level vocal isolation strength. <= 0 disables it for this request.",
    ),
    rms_threshold: Optional[float] = Query(
        None,
        description="Request-level RMS threshold. <= 0 disables threshold filtering for this request.",
    ),
):
    """
    专为 UE Async HTTP Request 设计
    Body 直接是音频文件原始二进制，language 写死中文。
    在送入 ASR 前，会对音频做轻量人声分离/语音增强 + 阈值过滤预处理。
    可通过 query 参数临时覆盖预处理强度：
      vocal_isolation_strength <= 0  表示关闭人声增强
      rms_threshold <= 0             表示关闭 RMS 阈值过滤
    
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

    preprocess_config = resolve_ue_preprocess_config(
        vocal_isolation_strength=vocal_isolation_strength,
        rms_threshold=rms_threshold,
    )
    if vocal_isolation_strength is not None or rms_threshold is not None:
        logger.info(
            "UE 请求覆盖预处理参数: vocal_isolation_strength=%s rms_threshold=%s -> vocal_enabled=%s vocal_strength=%.3f threshold_enabled=%s threshold_rms=%.6f",
            vocal_isolation_strength,
            rms_threshold,
            preprocess_config.vocal_isolation_enabled,
            preprocess_config.vocal_isolation_strength,
            preprocess_config.threshold_filter_enabled,
            preprocess_config.threshold_filter_rms,
        )

    try:
        # 1. 先解码请求体并保存原始导入音频，便于问题排查和听感对比。
        wav_raw, sr = await to_thread_limited(decode_sem, read_audio_file, file_bytes)
        await asyncio.to_thread(
            dump_audio_file,
            wav_raw,
            sr,
            "transcribe_ue_raw",
            "ue_request",
            enabled=EXPORT_INPUT_AUDIO,
        )

        # 2. 再执行 UE 专用预处理链：转单声道 -> 可选人声增强 -> 可选阈值过滤。
        wav = await to_thread_limited(decode_sem, preprocess_ue_audio_for_asr, wav_raw, sr, preprocess_config)
        if UE_EXPORT_PREPROCESSED_AUDIO:
            await asyncio.to_thread(
                dump_audio_file,
                wav,
                sr,
                "transcribe_ue_preprocessed",
                "ue_vocal_isolated",
                enabled=UE_EXPORT_PREPROCESSED_AUDIO,
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio: {e}")

    try:
        # 3. 最后将预处理后的音频送入 ASR 模型。
        async with infer_sem:
            results = await asyncio.to_thread(
                model.transcribe,
                audio=[(wav, sr)],
                language=["Chinese"],   # 写死中文
            )
        # 保留最近一次预处理结果，方便本地直接试听。
        sf.write("output.wav", wav, samplerate=sr)
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
