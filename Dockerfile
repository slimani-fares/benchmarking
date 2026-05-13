FROM registry.gitlab.inria.fr/magnet/declearn/declearn2/ci-python311:latest

WORKDIR /work/declearn-benchmarks
COPY benchmarks ./benchmarks

ENV TF_FORCE_GPU_ALLOW_GROWTH=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    DECLEARN_BENCH_FORCE_GPU=1 \
    BENCH_VENV=/venv

RUN /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir \
        "declearn[torch,tensorflow,websockets]==2.8.0" \
        "websockets<14.0" \
        asv cryptography

WORKDIR /work/declearn-benchmarks/benchmarks
CMD ["bash", "run_benchmarks.sh"]
