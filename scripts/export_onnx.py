#!/usr/bin/env python3
"""
Qwen3-ASR ONNX 导出说明
======================
Qwen3-ASR 的音频塔（audio tower）内部使用大量 Python 动态控制流：
  - chunk_lengths.tolist() 作为 split 参数
  - for input_feature, feature_len in zip(input_features, feature_lens)：逐样本循环 + Tensor 做切片
  - [torch.ones(length, ...) for length in feature_lens_after_cnn]：list comprehension 迭代 Tensor

这些模式与 TorchScript trace 和 torch.export 均不兼容，无法通过本脚本导出静态 ONNX。

推荐替代方案
-----------
1. torch.compile + TensorRT 后端（无需静态图，支持 Python 控制流）：
     pip install torch-tensorrt
     在推理代码中：
       model.thinker = torch.compile(model.thinker, backend="tensorrt",
                                      options={"enabled_precisions": {torch.float16}})

2. vLLM + TensorRT-LLM 后端（Qwen3-ASR 官方推荐，开箱即用）：
     vllm serve Qwen/Qwen3-ASR-1.7B --dtype float16

3. 直接使用 qwen-asr + vLLM（server.py 已实现，性能已足够高）：
     make -f Makefile up   # 或 make -f Makefile.tensorrt up BACKEND=llm
"""
import sys


def main():
    print(__doc__, file=sys.stderr)
    print(
        "错误: Qwen3-ASR 因音频塔存在大量动态 Python 控制流，\n"
        "      无法通过 TorchScript/torch.export 导出为静态 ONNX。\n"
        "      请参见上方说明，改用 torch.compile(backend='tensorrt')\n"
        "      或 vLLM TRT-LLM 后端。",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
