# Multi-stage build: builder pre-bakes models, runtime is lean
# Build: docker build -t whisper-transcribe:2.0 .
# Slim:  docker build --build-arg PREBAKE_ALL=0 -t whisper-transcribe:2.0-slim .
#
# Requires NVIDIA driver >= 570 for Blackwell (RTX 50xx) GPUs.

ARG BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04

# ---- Builder stage ----
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch nightly for Blackwell sm_120 support (stable 2.9 not yet released)
# Switch to --index-url https://download.pytorch.org/whl/cu128 once PyTorch 2.9 stable ships
RUN pip3 install --no-cache-dir --pre \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# cuDNN 9 for CUDA 12.8 — let PyTorch's bundled libs handle it
RUN pip3 install --no-cache-dir nvidia-cudnn-cu12==9.* && \
    find /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib -name "libcudnn*.so*" \
      -exec ln -sf {} /usr/local/lib/ \; 2>/dev/null || true

# Unpinned for Blackwell compatibility — pip resolves latest compatible versions
RUN pip3 install --no-cache-dir \
    ctranslate2 \
    faster-whisper \
    whisperx \
    pyannote.audio \
    huggingface_hub \
    hf_xet \
    soundfile numpy scipy matplotlib \
    loguru appdirs sympy unidecode \
    progressbar2 librosa timm torchlibrosa \
    fastapi "uvicorn[standard]" python-multipart && \
    pip3 install --no-cache-dir --no-deps audiosr deepfilternet

# Pre-bake models
ARG PREBAKE_ALL=1
ENV PREBAKE_ALL=${PREBAKE_ALL}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

RUN python3 <<'PYEOF'
import os
from huggingface_hub import hf_hub_download, list_repo_files
os.makedirs('/opt/whisper-models/large-v3', exist_ok=True)
for f in list_repo_files('Systran/faster-whisper-large-v3'):
    if f.endswith(('.bin', '.json')):
        path = hf_hub_download('Systran/faster-whisper-large-v3', f)
        os.system(f'cp {path} /opt/whisper-models/large-v3/')
        print(f'  {f}')
print('large-v3 downloaded.')
PYEOF

RUN if [ "$PREBAKE_ALL" = "1" ]; then \
    python3 -c "from faster_whisper import WhisperModel; print('Downloading Dutch model...')" && \
    ct2-transformers-converter \
        --model golesheed/whisper-large-v3-dutch \
        --output_dir /opt/whisper-models/nl-large-v3 \
        --quantization float16 \
        --copy_files tokenizer.json preprocessor_config.json || \
    echo "Dutch model conversion skipped (may need HF auth)"; \
    fi

RUN python3 <<'PYEOF'
import os
if os.environ.get('PREBAKE_ALL', '1') != '1':
    print('Skipping alignment model download (PREBAKE_ALL=0)')
    raise SystemExit(0)
from whisperx import load_align_model
for lang, name in [('nl', 'jonatasgrosman/wav2vec2-large-xlsr-53-dutch'), ('en', 'WAV2VEC2_ASR_BASE_960H')]:
    try:
        print(f'Downloading alignment model for {lang}...')
        load_align_model(language_code=lang, model_name=name, device='cpu')
    except Exception as e:
        print(f'  skipped {lang}: {e}')
PYEOF

# ---- Runtime stage ----
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --pre \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

RUN pip3 install --no-cache-dir nvidia-cudnn-cu12==9.* && \
    find /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib -name "libcudnn*.so*" \
      -exec ln -sf {} /usr/local/lib/ \; 2>/dev/null || true

RUN pip3 install --no-cache-dir \
    ctranslate2 \
    faster-whisper \
    whisperx \
    pyannote.audio \
    huggingface_hub \
    hf_xet \
    fastapi "uvicorn[standard]" python-multipart \
    soundfile numpy scipy matplotlib \
    loguru appdirs sympy unidecode \
    progressbar2 librosa timm torchlibrosa && \
    pip3 install --no-cache-dir --no-deps audiosr deepfilternet

COPY --from=builder /opt/whisper-models /opt/whisper-models
COPY pipeline/ /opt/pipeline/

ENV WHISPER_MODEL_DIR=/opt/whisper-models

RUN python3 -c "import faster_whisper, whisperx, torch, pyannote.audio; print('imports ok')"

WORKDIR /audio
EXPOSE 8000

ENTRYPOINT ["python3", "-m"]
CMD ["uvicorn", "pipeline.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
