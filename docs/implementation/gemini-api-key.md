# Getting a Gemini API key

Local-RAG uses the Google Gemini API for answer generation (see [setup.md](setup.md)). The free tier on **Google AI Studio** is enough for normal personal use of this app — the Gemini 2.5 Flash family (including the app's default, `gemini-2.5-flash-lite`) is free for both input and output, subject to per-minute and per-day rate limits.

This doc walks through getting a key, putting it where the app expects it, and verifying it works.

## 1. Prerequisites

- A **Google account** (personal Gmail is fine).
- A web browser.
- No billing account is required for the free tier. You'll see an option to "Set up Billing" later — you can ignore it unless you hit the free-tier limits and want to upgrade.

## 2. Create the key in Google AI Studio

1. Go to <https://aistudio.google.com/apikey>.
2. Sign in with your Google account if prompted.
3. If this is your first visit, accept the **Google AI Studio Terms of Service**.
4. Click **Create API key**.
5. When asked to associate the key with a Google Cloud project:
   - For personal use, accept the auto-created project (something like `Gemini API`).
   - If you already have a Cloud project you want to use, pick it from the dropdown.
6. The key is shown **once**. Copy it immediately — it looks like `AIzaSy…` (~39 chars).

> **Treat the key like a password.** Anyone with it can spend against your quota (and your billing account, if you ever attach one). Don't paste it into chats, screenshots, commit messages, or public repos.

## 3. Add the key to Local-RAG

From the repo root (`Local-RAG/`):

```powershell
Copy-Item .env.example .env
notepad .env
```

Edit the file so it looks like:

```
GEMINI_API_KEY=AIzaSy...your-real-key...
```

Save and close. `.env` is already in `.gitignore`, so it won't be committed.

**Alternative — shell environment variable** (per-session):

```powershell
$env:GEMINI_API_KEY = "AIzaSy...your-real-key..."
uv run streamlit run app.py
```

The app accepts either `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

## 4. Verify it works

From the repo root:

```powershell
uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('key length:', len(os.environ.get('GEMINI_API_KEY','')))"
```

You should see something like `key length: 39`. If you see `0`, the file isn't being read — check that it's named exactly `.env` (not `.env.txt`) and lives in the `Local-RAG/` folder.

Then run the app and ask a question:

```powershell
uv run streamlit run app.py
```

If the key is missing or invalid, the Streamlit page shows a clear error from `agent.py` and stops.

## 5. Free-tier limits (as of April 2026)

See <https://ai.google.dev/gemini-api/docs/pricing> for current numbers. At time of writing:

| Model | Free RPM | Free RPD |
|---|---|---|
| `gemini-2.5-flash-lite` *(default)* | ~15 | ~1000 |
| `gemini-2.5-flash` | ~10 | ~250 |
| `gemini-2.5-pro` | ~5 | ~25 |

A "request" here is one Gemini call. Each chat turn is **typically two calls** (one to decide whether to call the search tool, one to answer with the retrieved chunks), so plan your daily question budget accordingly.

If you hit a `429`, wait a minute, make sure `GEMINI_MODEL` isn't set to a tighter-quota model (the default `gemini-2.5-flash-lite` has the most generous free quota), or upgrade in AI Studio.

## 6. Privacy reminder

On the **free tier**, prompts and responses may be used by Google to improve their products. The retrieved chunks (~20 passages per question) are part of every prompt. If your PDFs are confidential, either:

- Upgrade to a paid Gemini tier in AI Studio (which opts your traffic out), or
- Switch the generation backend back to a local model (see git history for the previous Ollama-based `agent.py`).

The PDF *corpus* itself never leaves your machine — embeddings still run locally via Ollama.

## 7. Rotating or revoking a key

Back at <https://aistudio.google.com/apikey>:

- Click the key row to view it.
- Use **Delete** to revoke a leaked key, then create a new one and update `.env`.
- There's no "regenerate" button — revoke and recreate.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RuntimeError: GEMINI_API_KEY is not set` on app start | `.env` missing, mis-named, or in the wrong folder. Confirm `Get-ChildItem .env` from the repo root shows it. |
| `401 / 403 PERMISSION_DENIED` | Key was revoked or copied incompletely. Recreate. |
| `429 RESOURCE_EXHAUSTED` | Rate-limited. Wait, switch model, or upgrade. |
| Key works in `python -c` but Streamlit still errors | The Streamlit process started **before** you added the key. Stop it (Ctrl+C in its terminal) and re-run. |
