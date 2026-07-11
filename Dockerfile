# Qwen3-TTS pure-C inference engine — CPU-only container.
#
# Build:
#   docker build -t qwen3-tts .
#   docker build --build-arg SIMD=scalar -t qwen3-tts .   # pre-2013 x86 CPUs
#
# Run (models are downloaded on first start into /models — mount a volume to persist):
#   docker run -p 8080:8080 -v qwen3-tts-models:/models qwen3-tts
#
# Env:
#   MODEL      small | large | voice-design | base-small | base-large   (default: large)
#   PORT       HTTP port (default: 8080)
#   WORKERS    concurrent synthesis workers (default: 4)
#   BATCH_SIZE request batching, >=2 enables (default: 1)
#   QUANT      int8 | int4 | bf16   (default: int8; int4 is 1.7B-only)

FROM debian:bookworm-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev make libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# SIMD=auto -> portable -mavx2 -mfma baseline on x86 (see Makefile). On arm64 the
# Makefile uses -march=native, which under VMs/emulation misses dotprod (needed by
# the SDOT int8 kernels) — pin a portable armv8.2+dotprod baseline instead
# (M1-class, Graviton2+, Ampere all have it).
ARG SIMD=auto
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        make blas ARCH_FLAGS="-march=armv8.2-a+dotprod"; \
    else \
        make blas SIMD=$SIMD; \
    fi

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/qwen_tts /app/qwen_tts
# emotion/expressivity presets are loaded relative to the working directory
COPY presets /app/presets
COPY download_model.sh docker-entrypoint.sh /app/
RUN chmod +x /app/download_model.sh /app/docker-entrypoint.sh

ENV MODEL=large \
    MODELS_ROOT=/models \
    PORT=8080 \
    WORKERS=4 \
    BATCH_SIZE=1 \
    QUANT=int8

EXPOSE 8080
VOLUME /models

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS "http://127.0.0.1:${PORT}/v1/health" || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
