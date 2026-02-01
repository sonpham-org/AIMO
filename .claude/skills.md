# Claude Code Skills for AIMO Project

## Skill: Conversation Logging

**When to trigger:** After every significant discussion point, decision, or research finding in a session.

**What to do:**
1. Read `.claude/conversation_log.md` at the **start of every new session** to recall prior context
2. After each important exchange (research findings, architectural decisions, key insights, debugging breakthroughs), **append** to `.claude/conversation_log.md`
3. Use the format: date header → topic → bullet points with key details
4. Be specific — include model names, dataset sizes, URLs, scores, and technical parameters
5. At the end of a session (if the user says goodbye or the conversation naturally ends), write a session summary

**Why this exists:** The user wants continuity across sessions. Without this log, each session starts from scratch and past research/decisions are lost.

**File location:** `/home/son/GitHub/AIMO/.claude/conversation_log.md`

## Skill: Multi-Agent Sync via Conversation Log

**When to trigger:** When multiple Claude Code instances work on this project simultaneously (e.g., one on AMD machine, one on NVIDIA machine).

**What to do:**
1. **At session start:** `git pull origin main` to get latest conversation log
2. **Before reading conversation_log.md:** Always `git pull` first to get updates from other agents
3. **After writing to conversation_log.md:** Commit and push immediately:
   ```
   git add .claude/conversation_log.md
   git commit -m "Update conversation log"
   git push origin main
   ```
4. **Before making decisions that depend on other agents' work:** Pull and re-read the log
5. **Tag entries with machine/agent identity** when multiple agents are active, e.g.:
   ```
   ### [AMD-8060s] Session 4: ...
   ### [NVIDIA-4090] Session 4: ...
   ```

**Conflict resolution:** If `git push` fails due to conflicts, pull with rebase (`git pull --rebase origin main`), resolve any merge conflicts in the log (keep both entries), then push again.

**Why this exists:** The user runs Claude Code on multiple machines (AMD 8060s, NVIDIA 4090). The conversation log in git is the primary mechanism for agents to share context, decisions, and progress.

---

## Skill: Kaggle Dataset Upload

**When to trigger:** When uploading wheels, models, or data to Kaggle as a dataset.

**What to do:**
1. Export the API token: `export KAGGLE_API_TOKEN=<token from .env>`
2. Create `dataset-metadata.json` in the data directory with:
   ```json
   {
     "title": "Human-readable title",
     "id": "sonphamorg/slug-name",
     "licenses": [{"name": "Apache 2.0"}]
   }
   ```
3. Upload: `kaggle datasets create -p /path/to/dir/`
4. For updates: `kaggle datasets version -p /path/to/dir/ -m "version message"`

**Key details:**
- **NEVER use or modify `~/.kaggle/kaggle.json`** — it may contain stale/wrong credentials
- **ALWAYS** authenticate via: `export KAGGLE_API_TOKEN=<token from .env>` before any `kaggle` CLI command
- The token is stored in `.env` (gitignored) as `KAGGLE_API_TOKEN=KGAT_...`
- Load it with: `export $(grep KAGGLE_API_TOKEN .env | xargs)`
- Max dataset size: ~20GB
- Kaggle username: `sonphamorg`

---

## Skill: vLLM Wheel Management for Kaggle

**When to trigger:** When preparing vLLM for Kaggle offline competition environments.

**What to do:**
1. Use `pip download vllm==<version> --dest <dir> --python-version 3.12 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 --platform linux_x86_64 --only-binary :all:` to download all wheels
2. Create slim version by excluding packages Kaggle already has (torch, nvidia-*, triton, numpy, pillow, etc.)
3. Upload as Kaggle dataset
4. In notebook: `pip install --no-index --find-links /kaggle/input/<dataset>/ vllm==<version>`

**Version compatibility (as of 2026-02):**
- Kaggle has: Python 3.12, CUDA 12.5+ (nvcc), cuda-python 12.9.4
- **Full dataset approach:** vLLM 0.15.0 + torch 2.9.1+cu129 + all deps (163 wheels, 5.3GB)
  - Tested on RTX 4090, Python 3.12, CUDA 12.9 — works perfectly
  - Dataset: `sonphamorg/vllm-wheels-py312-cu129`
  - Install: `pip install --no-index --find-links /kaggle/input/vllm-wheels-py312-cu129/wheels/ vllm`
- **Slim dataset (older):** vLLM 0.8.0 (CUDA 12.4, torch 2.6.0)
  - Avoid vLLM ≥0.8.5 if relying on Kaggle's pre-installed torch
  - Dataset: `sonphamorg/vllm-wheels-cp312` — slim set (~668MB, 104 wheels, no torch/nvidia)
  - `sonphamorg/vllm-offline-install-cp312-fix` — old minimal set (msgspec + xgrammar only)

---

## How to Use This File

This `skills.md` file should be referenced by placing it in the project's `CLAUDE.md` or by the user reminding Claude to check `.claude/` at session start. The conversation log serves as persistent memory for this project.
