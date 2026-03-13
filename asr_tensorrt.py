"""
TensorRT 后端：加载 ONNX 导出的 Qwen3-ASR 引擎，提供与 server 兼容的 transcribe 接口。
仅支持批量 /transcribe，流式 /transcribe-streaming 使用 TensorRT 时不支持（可回退到逐段整段推理）。
依赖: tensorrt, pycuda, transformers, torch, numpy
"""
from __future__ import annotations

import os
from typing import List, Tuple, Any, Optional

import numpy as np
import torch

try:
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None
    cuda = None

try:
    from transformers import AutoProcessor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoProcessor = None


class TranscribeResult:
    """与 server 期望一致的转录结果"""
    __slots__ = ("text", "language")

    def __init__(self, text: str, language: Optional[str] = None):
        self.text = text
        self.language = language or ""


def _decode_logits_greedy(logits: np.ndarray, tokenizer: Any) -> str:
    """logits [seq_len, vocab_size] -> 贪心解码 -> 文本"""
    ids = np.argmax(logits, axis=-1)
    # 转 list，去掉 padding/special
    ids = ids.flatten().tolist()
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


class Qwen3ASRTensorRT:
    """
    使用 TensorRT 引擎的 Qwen3-ASR 推理封装。
    接口与 qwen_asr 的 transcribe 兼容：transcribe(audio=..., language=..., return_time_stamps=False)。
    """

    def __init__(
        self,
        engine_path: str,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
    ):
        if not TRT_AVAILABLE:
            raise RuntimeError("tensorrt or pycuda not installed")
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed (need AutoProcessor)")

        self.engine_path = os.path.abspath(engine_path)
        self.model_id = model_id
        self.device = device
        self._logger = trt.Logger(trt.Logger.WARNING)

        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(self._logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._tokenizer = getattr(
            self._processor, "tokenizer", None
        ) or self._processor.feature_extractor  # 若 tokenizer 在 processor 内
        if not hasattr(self._tokenizer, "decode"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        self._stream = cuda.Stream()
        self._sample_rate = 16000

        # TensorRT 8 使用 binding 索引
        self._idx_input_f = self._engine.get_binding_index("input_features")
        self._idx_input_m = self._engine.get_binding_index("attention_mask")
        self._idx_logits = self._engine.get_binding_index("logits")

    def _run_engine(
        self,
        input_features: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """运行 TensorRT 引擎，返回 logits。"""
        self._context.set_binding_shape(self._idx_input_f, input_features.shape)
        self._context.set_binding_shape(self._idx_input_m, attention_mask.shape)

        out_shape = tuple(self._context.get_binding_shape(self._idx_logits))
        out_size = int(np.prod(out_shape))
        out_dtype = np.float32
        logits_nbytes = out_size * np.dtype(out_dtype).itemsize

        d_input_f = cuda.mem_alloc(input_features.nbytes)
        d_input_m = cuda.mem_alloc(attention_mask.nbytes)
        d_logits = cuda.mem_alloc(logits_nbytes)

        cuda.memcpy_htod_async(d_input_f, input_features, self._stream)
        cuda.memcpy_htod_async(d_input_m, attention_mask, self._stream)

        n = self._engine.num_bindings
        bindings = [0] * n
        bindings[self._idx_input_f] = int(d_input_f)
        bindings[self._idx_input_m] = int(d_input_m)
        bindings[self._idx_logits] = int(d_logits)

        self._context.execute_async_v2(bindings, self._stream.handle)
        self._stream.synchronize()

        logits = np.empty((out_size,), dtype=out_dtype)
        cuda.memcpy_dtoh_async(logits, d_logits, self._stream)
        self._stream.synchronize()

        return logits.reshape(out_shape)

    def transcribe(
        self,
        audio: List[Tuple[np.ndarray, int]],
        language: Optional[List[Optional[str]]] = None,
        return_time_stamps: bool = False,
    ) -> List[TranscribeResult]:
        """
        批量转录。audio 为 (samples, sample_rate) 列表。
        返回与 server 兼容的 TranscribeResult 列表（.text, .language）。
        """
        if language is None:
            language = [None] * len(audio)
        results = []
        for i, (wav, sr) in enumerate(audio):
            lang = language[i] if i < len(language) else None
            text = self._transcribe_one(wav, sr, lang)
            results.append(TranscribeResult(text=text, language=lang or ""))
        return results

    def _transcribe_one(
        self,
        wav: np.ndarray,
        sr: int,
        language: Optional[str],
    ) -> str:
        """单条音频转录。"""
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        if sr != self._sample_rate:
            try:
                from scipy import signal
                num = int(len(wav) * self._sample_rate / sr)
                wav = signal.resample(wav, num).astype(np.float32)
            except ImportError:
                # 无 scipy 时仅做长度近似（截断或零填充）
                target_len = int(len(wav) * self._sample_rate / sr)
                if len(wav) >= target_len:
                    wav = wav[:target_len]
                else:
                    wav = np.pad(wav, (0, target_len - len(wav)))

        inputs = self._processor(
            audio=wav,
            sampling_rate=self._sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.numpy().astype(np.float32)
        attention_mask = inputs.attention_mask.numpy().astype(np.int32)

        logits = self._run_engine(input_features, attention_mask)
        # logits 可能 [1, seq_len, vocab_size]，取第一个 batch
        if logits.ndim == 3:
            logits = logits[0]
        return _decode_logits_greedy(logits, self._tokenizer)

    # 流式接口：TensorRT 方案下用“整段推理”模拟，不实现真正流式
    def init_streaming_state(
        self,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> Any:
        """占位：返回一个可持有缓冲的状态对象，用于兼容 server 流式逻辑。"""
        class State:
            def __init__(self):
                self.text = ""
                self.language = ""
                self.buf = []
                self.buf_n = 0
        return State()

    def streaming_transcribe(self, chunk: np.ndarray, state: Any) -> None:
        """将 chunk 加入 state 缓冲，不立即推理。"""
        state.buf.append(chunk)
        state.buf_n += len(chunk)

    def finish_streaming_transcribe(self, state: Any) -> None:
        """对缓冲整段做一次转录，写入 state.text / state.language。"""
        if state.buf_n == 0:
            state.text = ""
            state.language = ""
            return
        wav = np.concatenate(state.buf, axis=0) if len(state.buf) > 1 else state.buf[0]
        state.text = self._transcribe_one(wav, self._sample_rate, None)
        state.language = ""
