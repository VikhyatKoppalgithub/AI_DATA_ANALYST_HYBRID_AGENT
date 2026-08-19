# Deploying the live demo to Streamlit Community Cloud

Free, and no card. Takes about ten minutes.

## Why this and not a local model

The project runs on Ollama at $0.00 per query, and that is still the default
everywhere except this deployment. But no free host will run a local LLM:
Hugging Face made Docker Spaces PRO-only, and Streamlit Community Cloud has
neither Ollama nor the RAM for a 1GB model. So the demo points the same pipeline
at Gemini's free tier — 15 requests/minute, 1,500/day, genuinely $0.

This is worth being precise about in an interview, because it is the provider
abstraction earning its keep rather than being asserted: one `Provider`
implementation swapped, no change anywhere in the analysis code, and the
deployed demo runs identical arithmetic. The app says so in a banner rather than
quietly presenting a hosted model as the local one.

## 1. Get a Gemini API key

<https://aistudio.google.com/apikey> → **Create API key**. The free tier needs no
billing account.

**Do not paste it into a file in this repo.** It goes into Streamlit's secrets
box in step 3, which is not in version control. I never see it either.

## 2. Create the app

<https://share.streamlit.io> → sign in with GitHub → **Create app** →
**Deploy a public app from a repo**:

| field | value |
|---|---|
| Repository | `VikhyatKoppalgithub/AI_DATA_ANALYST_HYBRID_AGENT` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL | pick anything free, e.g. `ai-data-analyst` |

## 3. Add the key before you deploy

Click **Advanced settings** → **Secrets**, and paste exactly this, with your own
key:

```toml
GEMINI_API_KEY = "paste-your-key-here"
```

Then **Deploy**. First build takes 3–5 minutes while it installs
`requirements.txt`.

That single secret is all that is needed. The app checks for a Gemini key and
switches providers on its own — no flags, no code edits. Without it, it falls
back to looking for a local Ollama, which is what you want on your laptop.

## 4. Check it

| what | expected |
|---|---|
| Sidebar | green, `gemini-2.5-flash ready (Gemini free tier, hosted)` |
| Banner | cloud icon, explaining the hosted swap |
| **Data profile** tab | renders instantly, needs no model |
| *"Why did revenue drop in March?"* | verified answer, `Laptop x West −11.08 pp`, six green checks |
| *"What is the correlation between units and unit_price?"* | code route, amber unverified banner, generated Python shown |

## Fixes for what may break

| symptom | cause | fix |
|---|---|---|
| Sidebar red, `no model ...` | Google renamed or retired it | the error now lists models your key can use — copy one into Secrets as `GEMINI_MODEL` |
| Red, `rejected the API key` | key wrong, or the Generative Language API is not enabled on it | regenerate at aistudio.google.com |
| `rate limit hit` under traffic | free tier is 15 req/min; each verified answer is 3 calls | expected at ~5 questions/minute; wait, or upgrade the key |
| Code-route questions error out | the host blocks subprocesses or `rlimit` | add `ANALYST_NO_CODEGEN = "1"` to Secrets — the verified path keeps working |
| App sleeps after inactivity | free tier behaviour | first visitor waits ~30s for wake-up; unavoidable |

## What is untested

The Gemini provider has 18 unit tests, all of which substitute the network, so
its request shape, role mapping, schema cleaning, and error handling are
verified — but **no real call to Google has ever been made from this code**. I
have no API key and should not have one. The first genuine request happens on
your deploy, so budget one debugging pass; the table above covers what I would
expect.

## Local use is unchanged

Without `GEMINI_API_KEY` set, the app looks for Ollama exactly as before. Nothing
about the local path, the CLI, or the eval suite moved.
