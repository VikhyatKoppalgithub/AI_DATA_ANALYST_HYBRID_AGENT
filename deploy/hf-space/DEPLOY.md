# Deploying the live demo to a Hugging Face Space

> **Superseded — this route now costs $9/month.** Hugging Face has made Docker
> and Gradio Spaces PRO-only; only Static Spaces (no compute, no Python) remain
> free, so this Dockerfile cannot run on a free account. It is kept because it is
> the only way to demo the project *as designed* — a local model, no API key —
> and because it works unchanged if you ever take a PRO plan.
>
> The free route is `deploy/streamlit/DEPLOY.md`, which trades the local model
> for Gemini's free tier.


Three files go in the Space; the app itself is cloned from GitHub at build time,
so the Space repo stays small and a rebuild picks up the latest `main`.

## Before you start — is this demo worth linking?

Measured, not guessed. On an M4 with Metal, `qwen2.5-coder:1.5b` runs at
**441 tok/s** prompt eval and **78.7 tok/s** generation, and a full verified
answer takes **~6 seconds**. A free Space has 2 vCPU and no GPU, where the same
model realistically manages 30–80 tok/s prompt and 8–15 tok/s generation.

A verified answer is three model calls — routing, planning, narration — costing
roughly 3,700 prompt tokens and 400 generated. That extrapolates to:

| | local (Metal) | HF free CPU |
|---|---|---|
| per question | ~6s | **~75–175s** |
| plus cold start | — | 30–60s if the Space is asleep |

So a visitor may wait two to three minutes for their first answer. Decide
knowingly:

- **The Data profile tab needs no model** and renders instantly. It is the
  fastest way to show what the agent sends the LLM, and it always works.
- If the wait proves too long, the fix is to ship precomputed answers for the
  two bundled example questions — labelled as cached — so the common path is
  instant and only custom questions pay for inference.
- Upgrading the Space to paid CPU or GPU hardware removes the problem and
  removes the "$0.00" claim with it.

## Steps

**1. Create the Space.** On huggingface.co → *New Space*:

| field | value |
|---|---|
| Owner | your account |
| Space name | `ai-data-analyst` |
| License | MIT |
| SDK | **Docker** → *Blank* |
| Hardware | CPU basic (free) |
| Visibility | Public |

You have to do this part — I can't create accounts.

**2. Push these three files** to the Space repo. `README.md` must stay at the
root with its YAML frontmatter intact; that frontmatter is what tells HF to use
Docker and to expose port 7860.

```bash
git clone https://huggingface.co/spaces/<your-username>/ai-data-analyst
cd ai-data-analyst
cp /path/to/ai-data-analyst/deploy/hf-space/{Dockerfile,start.sh,README.md} .
git add -A && git commit -m "AI Data Analyst live demo" && git push
```

**3. Watch the first build.** It takes roughly 10–15 minutes — it installs
Ollama, clones the app, and bakes the ~1GB model into the image. The build log
is on the Space page. Failures to expect on a first run:

| symptom | cause | fix |
|---|---|---|
| `ollama: command not found` | install script put the binary somewhere unexpected | add `ENV PATH=/usr/local/bin:$PATH` before the pull step |
| pull step hangs then fails | server not up before `ollama pull` ran | raise the retry count in the Dockerfile's wait loop |
| Space builds, page never loads | port mismatch | `app_port` in README.md frontmatter must be 7860, matching `start.sh` |
| `permission denied` on models | `OLLAMA_MODELS` not writable by uid 1000 | confirm the `ENV OLLAMA_MODELS=/home/user/...` line survived |

**4. Add the link to the portfolio.** Once the Space is live, in
`vikhyat-portfolio/content/projects.ts`, the `ai-data-analyst` entry:

```ts
    links: {
      github: "https://github.com/VikhyatKoppalgithub/AI_DATA_ANALYST_HYBRID_AGENT",
      demo: "https://huggingface.co/spaces/<your-username>/ai-data-analyst",
    },
```

The card already renders a "Live demo" chip whenever `demo` is set — no
component changes needed.

## Updating it later

The Dockerfile clones `main` at build time, so a rebuild picks up new commits.
Force one by bumping the cache-busting arg in the Space's *Settings → Factory
rebuild*, or push any commit to the Space repo.

## What is untested

I could not build this image — Docker is not installed on the machine it was
written on. The Dockerfile and entrypoint are written carefully and the shell
syntax is checked, but the first real build happens on HF. Budget one debugging
pass; the table above covers the failures worth anticipating.
