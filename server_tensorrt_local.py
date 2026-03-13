"""
本地模型加载版入口：
- 复用 `server_tensorrt.py` 中已经实现好的 FastAPI 应用和推理逻辑
- 依赖外部环境变量控制模型路径（ASR_MODEL_NAME、HF_HOME 等）

在 Docker 场景下，配合 `Makefile.tensorrt.local` 与 `Dockerfile.tensorrt.local`
即可实现「只从本地挂载的 hf_models 目录加载模型，不访问网络」。
"""

from server_tensorrt import app  # FastAPI 实例

