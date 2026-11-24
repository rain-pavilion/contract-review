# 智律契灵 v2 —— 云端部署镜像（CPU 即可，无需 GPU）
#
# 为什么是 slim 而不是 alpine：sklearn / pymupdf / onnxruntime 都有 manylinux 轮子，
# alpine 的 musl 会逼着从源码编译，镜像构建从 3 分钟变成 40 分钟。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ZLQL_HOST=0.0.0.0 \
    ZLQL_PORT=8000 \
    PORT=8000

WORKDIR /app

# 先装依赖，利用层缓存：只改代码时不会重跑 pip
COPY deploy/requirements-deploy.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 只拷运行期需要的东西：内核代码 + 法条库 + 训练出的六个权重 JSON
COPY zhilv_qiling_v2/ /app/zhilv_qiling_v2/
COPY data/法条结构化数据.jsonl /app/data/法条结构化数据.jsonl
COPY data/qualification_kg.jsonl /app/data/qualification_kg.jsonl
COPY samples/ /app/samples/
COPY tools/acceptance_annotations.py tools/acceptance_common.py /app/tools/

# 权重与大文件不进镜像（见 .dockerignore）：预训练模型 3~16 GB，走挂载或远程推理

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=4).status==200 else 1)"

CMD ["python", "-m", "zhilv_qiling_v2.api", "--port", "8000"]
