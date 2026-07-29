# V101.36 Test Report

Date: 2026-07-29

## Focused regression suite

Command:

```bash
PYTHONPATH=. pytest -q tests/test_v101_36_measurement_reliability.py
```

Result:

```text
9 passed
```

Covered:

- ASLR selected raised-leg hip rather than shared pelvis midpoint
- ASLR cross-label chain geometry
- shoulder 170 degrees classified yellow
- shoulder green begins at 175 degrees
- image-fingerprint-aware submission
- ephemeral QA handling
- posture low-confidence rejection
- squat coherent visible-side chain
- same-image reuse versus different-image retake

## Complete inherited suite

Current supplied backend baseline:

```text
34 passed, 20 failed
```

Complete backend with V101.36 patch plus nine new tests:

```text
43 passed, 20 failed
```

The patch introduced no additional inherited-suite failures. The remaining 20 failures are pre-existing stale architecture/version assertions in the supplied repository.
