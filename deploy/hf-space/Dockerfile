# Hugging Face Space — AI Data Analyst, running fully local on CPU.
#
# The point of this project is that no API key is involved and inference costs
# nothing, so the demo has to preserve that or it is advertising something the
# code does not do. That rules out every hosted-model shortcut and leaves one
# option: run Ollama inside the Space.
#
# The 14B used for the published eval numbers needs ~9GB and a GPU to be
# tolerable. A free CPU Space has 16GB RAM and 2 vCPU, so the demo runs
# qwen2.5-coder:1.5b instead — the same model the eval suite already scores at
# 80%, which means the demo's weaker answers are a documented property rather
# than an unexplained disappointment.
#
# The app itself is cloned from GitHub at build time rather than committed here,
# so this Space stays three small files and a rebuild picks up the latest main.

FROM python:3.11-slim

# procps supplies pkill, used to stop the temporary server after baking the
# model. git is needed for the clone below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git procps \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

# HF Spaces convention: run as uid 1000 rather than root.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/home/user/.ollama/models \
    ANALYST_MODEL=qwen2.5-coder:1.5b \
    ANALYST_DEMO=1 \
    PYTHONUNBUFFERED=1

USER user
WORKDIR /home/user/app

ARG REPO=https://github.com/VikhyatKoppalgithub/AI_DATA_ANALYST_HYBRID_AGENT.git
# Invalidate this layer to pick up new commits: bump with --build-arg CACHEBUST=$(date +%s)
ARG CACHEBUST=1
RUN git clone --depth 1 "$REPO" .

RUN pip install --no-cache-dir --user -e .

# Bake the weights into the image. Pulling at container start instead would make
# every cold start pay a ~1GB download, which on a Space that sleeps is most
# starts. `ollama serve` has no daemon mode, so it is backgrounded, polled until
# it answers, used, then killed.
RUN set -eux; \
    ollama serve & \
    for i in $(seq 1 60); do \
        curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; \
        sleep 1; \
    done; \
    ollama pull qwen2.5-coder:1.5b; \
    pkill -f "ollama serve" || true; \
    sleep 2

COPY --chown=user start.sh /home/user/start.sh

# Must match `app_port` in README.md's frontmatter.
EXPOSE 7860

CMD ["bash", "/home/user/start.sh"]
