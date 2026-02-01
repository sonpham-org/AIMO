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

## How to Use This File

This `skills.md` file should be referenced by placing it in the project's `CLAUDE.md` or by the user reminding Claude to check `.claude/` at session start. The conversation log serves as persistent memory for this project.
