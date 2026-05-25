# Research Materials

This directory stores research inputs and project-specific synthesis used to
plan implementation and experiments.

## Layout

```text
research/
  README.md              # This policy
  reports/               # LLM-generated or manually authored surveys
  notes/                 # Short notes for papers selected for use
  ideas/                 # Scoped issue candidates derived from research
  catalog.yaml           # Verified metadata index (added when populated)
  pdfs/                  # Local PDF cache; ignored by Git
```

## Tracking Policy

- Track Markdown reports, paper notes, and verified metadata in Git.
- Store downloaded PDFs under `research/pdfs/`; this directory is ignored.
- Record a stable identifier such as DOI, arXiv ID, ACL Anthology ID, or
  OpenAlex ID in metadata or notes instead of relying on a local PDF path.
- Treat generated research reports as leads until cited papers and
  publication status have been checked.

## Workflow

1. Save a generated survey under `reports/`.
2. Shortlist papers that directly affect claims, implementation, or evaluation.
3. Verify bibliographic metadata and record it in `catalog.yaml`.
4. Download only useful PDFs to `pdfs/` and write notes for papers that inform
   an issue or design decision.
5. Draft actionable, scoped ideas under `ideas/`, with references and
   acceptance criteria.
6. Promote accepted candidates into GitHub issues.
