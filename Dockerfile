# 使用 relext:dz-cu12 运行时作为父镜像
# relext:dz-cu12 的父镜像是 registry.baidubce.com/paddlepaddle/paddle:3.0.0b1-gpu-cuda12.3-cudnn9.0-trt8.6

FROM relext:dz-cu12

LABEL maintainer="dz" \
      version="relextapi:dz-cu12" \
      description="This is a relext api Docker image for project relext. Source code: https://github.com/shibing624/relext " \
      app="relext_api"
# 设置工作目录
WORKDIR /home/relext-src/examples

# 将当前目录内容复制到容器的/app中
ADD relext_api.py /home/relext-src/examples/

# 安装程序需要的包
#RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置默认端口号（可以在启动容器时覆盖）
ENV RELEXT_PORT=38000
# 运行时监听的端口
EXPOSE 38000

# 设置 RELEXT_WORKER_COUNT 模型并发数，越大小浩资源越多，负载能力越强
ENV RELEXT_WORKER_COUNT=1

# 运行app.py时的命令及其参数
#CMD ["sh", "-c", "uvicorn relext_api:app --host 0.0.0.0 --port $RELEXT_PORT --workers ${RELEXT_WORKER_COUNT:-1}"]
#CMD ["sh", "-c", "uvicorn relext_api:app --host 0.0.0.0 --port $RELEXT_PORT --workers ${RELEXT_WORKER_COUNT:-1}"]
#CMD ["uvicorn relext_api:app --host 0.0.0.0  --port $RELEXT_PORT --reload --workers ${RELEXT_WORKER_COUNT:-1}" ]
CMD ["sh", "-c", "python relext_api.py"]
