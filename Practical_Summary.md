# MLOps Practical Exam — Submission Summary

**Repository:** `Yashomati-Baviskar/MLOps-exam`

## Branch & Practical Overview

| Member | Practical Focus | Branch Name | Remote Status |
| :--- | :--- | :--- | :--- |
| Member 1 | Git version control & repository workflow | `member-1-practical-2` | Pushed to `origin` |
| Member 2 | DVC data version control | `member-2-practical-3` | Pushed to `origin` |
| Member 3 | Wine DVC pipeline (`wine-practical-4-member-3`) | `member-3-practical-4` | Pushed to `origin` |
| Member 4 | Feast feature store & online retrieval (`wine-practical-5-member-4`) | `member-4-practical-5` | Pushed to `origin` |
| Member 5 | MLflow model & experiment tracking (`wine-practical-6-member-5`) | `member-5-practical-6` | Pushed to `origin` |
| Member 6 | Final review, audit & summary | `member-6-extra` | Active / Finalized |

## Key Engineering & Troubleshooting Notes

- **Ignore Rules:** Virtual environments (`.venv/`), Python caches (`__pycache__/`, `*.pyc`), DVC local remotes, Feast SQLite files (`registry.db`, `online_store.db`), and MLflow artifacts/databases (`mlruns/`, `mlartifacts/`, `mlflow.db`) excluded via `.gitignore`.
- **Large Push Reliability:** Configured `http.postBuffer` (`524288000`) and removed `.venv/` directories to prevent HTTP timeouts on GitHub.
- **Tooling Fixes:** Applied Feast entity/feature view definitions, ran incremental materialization and online retrieval, and passed `skops_trusted_types=["sklearn.tree._tree.Tree"]` when logging scikit-learn tree models to MLflow.