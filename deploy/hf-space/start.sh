#!/usr/bin/env bash
# Space entrypoint: bring Ollama up, then hand the container to Streamlit.
#
# Two processes, no supervisor. Ollama is backgrounded and Streamlit runs in the
# foreground via exec, so Streamlit owns PID 1's signals and the container dies
# when it dies. If Ollama falls over mid-session the app degrades honestly — its
# health() check reports the provider unreachable and the non-model tabs still
# work — which is better than a supervisor silently restarting under the user.
set -euo pipefail

echo "starting ollama…"
ollama serve &
OLLAMA_PID=$!

# Poll rather than sleep a fixed amount: cold start on a free CPU Space is
# unpredictable, and a fixed wait is either wasteful or too short.
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "ollama up after ${i}s"
        break
    fi
    if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "ollama exited during startup" >&2
        exit 1
    fi
    sleep 1
done

# Load the weights now so the first visitor doesn't pay for it. Failure here is
# not fatal — the app surfaces an unreachable provider rather than 500ing.
echo "warming ${ANALYST_MODEL:-qwen2.5-coder:1.5b}…"
curl -sf http://127.0.0.1:11434/api/generate \
    -d "{\"model\":\"${ANALYST_MODEL:-qwen2.5-coder:1.5b}\",\"prompt\":\"ok\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
    >/dev/null 2>&1 || echo "warm-up call failed; continuing" >&2

echo "starting streamlit…"
# maxUploadSize is capped well below Streamlit's 200MB default: this is a public
# demo on 16GB shared with a model, and profiling a huge upload would evict it.
exec streamlit run app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.maxUploadSize 20 \
    --browser.gatherUsageStats false
