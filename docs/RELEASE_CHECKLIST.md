# Release checklist

Before uploading an anonymous artifact:

1. Run `medflow validate` for every configuration and ensure the tuple audit has no non-finite values.
2. Run a CPU smoke test and the unit tests in the parent workspace.
3. Re-run at least the MIMIC mortality and APAVA configurations from empty `runs/` directories, retaining all manifests.
4. Regenerate paper tables from the metric CSVs rather than hand-copying values.
5. Remove all raw-data paths, subject identifiers, checkpoints, outputs, editor metadata, and author identity from the artifact.
6. Replace the placeholder `LICENSE` with the authors' final project licence;
   retain `third_party/MICROSOFT_MIT_LICENSE.txt` and `third_party/NOTICE.md`.
7. State clearly whether the artifact reports one generator run or independent generator repetitions.
8. Run `python scripts/audit_release.py` immediately before `git add .`.

The public package should contain code, configurations, documentation, and a toy/synthetic test fixture only. It must not contain clinical tuples or any re-identifiable data.
