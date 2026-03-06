# dfm_scraping — AGENTS.md

## Purpose
This file defines local instructions for Codex when working in this repository.

## Guardrails
- Do not hand-edit or reformat large artifact JSON files (`posts*.json`, `authors.json`, `extractions*.json`) unless explicitly requested.
- do not change any of the posts files like posts_enriched, posts_raw, etc.
- don't write to the iterations folder; only read from there

## Style / conventions
- Python: keep existing formatting style; do not add new formatting/lint tooling unless asked.

## project context: 
**Research question:** How can undocumented and tribal machining knowledge be collected and structured to improve the available knowledge base for AI-powered DFM interventions?

Machinists accumulate enormous practical knowledge about what designs are easy or hard to machine — but this knowledge lives in forum posts, not textbooks. The goal is to extract it, structure it, and make it queryable for DFM (design for manufacturability) feedback systems.

**Target use case:** DFM feedback for CNC machined parts — primarily small-scale projects (student, makerspace, academic lab, small company) where established workflows aren't set up.

**Novelty:**
- Advances DFM beyond rule-based approaches to include shop context and practical experience
- Builds a structured, AI-readable knowledge base of DFM heuristics, risks, and principles
- Establishes a general NLP method for capturing specialized design knowledge from forums

## Other Notes

when running python files like extract.py and iterate.py, you need to make sure you've activated the virtual environment.
- Create/activate venv (Windows PowerShell):
  - `.\venv\Scripts\Activate.ps1`


## ADHERENCE TO SCOPE
- Do NOT edit any files in the root directory or parent directories.
- If a task requires editing outside this scope, you MUST stop and ask for permission.
- do not edit or delete from the folder with a name starting with "BACKUP"

## SAFETY PROTOCOLS (No Git detected)
- Before making significant changes to any file, create a backup copy named 'filename.py.bak' in the same directory.
- Never delete files (rm/Remove-Item); only move them to a ./trash/ folder if necessary.
