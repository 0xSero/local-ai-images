FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
LABEL org.opencontainers.image.title="Step-5 Spark runtime foundation" \
      org.opencontainers.image.description="Experimental ARM64 dependency image; serving acceptance pending. No model weights or user data."
ENV DEBIAN_FRONTEND=noninteractive VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    EXL3_INT8_GEMV=0 TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv python3-dev build-essential git ninja-build libnuma1 libibverbs1 librdmacm1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
RUN git clone https://github.com/turboderp-org/exllamav3.git /opt/exllamav3 \
    && git -C /opt/exllamav3 checkout --detach d3739fd393337b1ff4d6c2a342b12f0c87a9592f \
    && EXLLAMA_NOCOMPILE=1 pip install --no-cache-dir --no-build-isolation --no-deps /opt/exllamav3 \
    && rm -rf /opt/exllamav3/.git
COPY entrypoint.py /opt/entrypoint.py
WORKDIR /runtime
ENTRYPOINT ["python", "/opt/entrypoint.py"]
CMD ["--check"]
