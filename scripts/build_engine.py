#!/usr/bin/env python3
"""
从 ONNX 构建 TensorRT 引擎（FP16，可选 INT8 校准）。
依赖: TensorRT 已安装并设置 LD_LIBRARY_PATH/PYTHONPATH；可选 pycuda（仅 INT8 时需要）。
用法:
  python scripts/build_engine.py --onnx qwen3_asr_1.7b_sim.onnx --engine qwen3_asr_1.7b.engine
  python scripts/build_engine.py --onnx qwen3_asr_1.7b_sim.onnx --engine qwen3_asr_1.7b.engine --int8  # INT8 需 pycuda
"""
import argparse
import os
import tensorrt as trt


def build_fp16(onnx_path: str, engine_path: str, workspace_gb: int = 4):
    """构建 FP16 TensorRT 引擎，动态 shape。"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.max_workspace_size = workspace_gb * (1 << 30)

    profile = builder.create_optimization_profile()
    # 与文档一致：时间维 min/opt/max
    profile.set_shape(
        "input_features",
        min=(1, 80, 100),
        opt=(1, 80, 1000),
        max=(1, 80, 2000),
    )
    profile.set_shape(
        "attention_mask",
        min=(1, 100),
        opt=(1, 1000),
        max=(1, 2000),
    )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Engine saved to {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", default="qwen3_asr_1.7b_sim.onnx", help="Input ONNX path")
    parser.add_argument("--engine", default="qwen3_asr_1.7b.engine", help="Output engine path")
    parser.add_argument("--workspace_gb", type=int, default=4, help="Workspace size in GB")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 (requires pycuda + calibrator)")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(f"ONNX not found: {args.onnx}")

    if args.int8:
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import numpy as np
        except ImportError:
            raise ImportError("INT8 requires pycuda: pip install pycuda")

        # 简单 INT8 校准器：随机数据，与文档思路一致
        class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, cache_file="qwen3_asr_calib.cache"):
                super().__init__()
                self.cache_file = cache_file
                self.idx = 0
                self.data = []
                for _ in range(100):
                    length = int(np.random.randint(100, 2001))
                    self.data.append((
                        np.random.randn(1, 80, length).astype(np.float32),
                        np.ones((1, length), dtype=np.int32),
                    ))

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                if self.idx >= len(self.data):
                    return None
                feat, mask = self.data[self.idx]
                self.idx += 1
                d_feat = cuda.mem_alloc(feat.nbytes)
                d_mask = cuda.mem_alloc(mask.nbytes)
                cuda.memcpy_htod(d_feat, feat)
                cuda.memcpy_htod(d_mask, mask)
                return [int(d_feat), int(d_mask)]

            def read_calibration_cache(self):
                if os.path.isfile(self.cache_file):
                    with open(self.cache_file, "rb") as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        with open(args.onnx, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        config.max_workspace_size = args.workspace_gb * (1 << 30)
        config.int8_calibrator = SimpleCalibrator()

        profile = builder.create_optimization_profile()
        profile.set_shape("input_features", (1, 80, 100), (1, 80, 1000), (1, 80, 2000))
        profile.set_shape("attention_mask", (1, 100), (1, 1000), (1, 2000))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build failed (INT8)")
        with open(args.engine, "wb") as f:
            f.write(serialized)
        print(f"INT8 engine saved to {args.engine}")
    else:
        build_fp16(args.onnx, args.engine, args.workspace_gb)


if __name__ == "__main__":
    main()
