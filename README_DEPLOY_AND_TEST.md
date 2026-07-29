# FlexiLab Backend Measurement Reliability Patch V101.36.0

## Purpose

This patch corrects the measurement regression found during ASLR and shoulder testing, prevents a new retake photo from receiving an older cached result, and adds per-stage timing data for performance profiling.

This is a screening-engine refinement, not a claim of medical-device or diagnostic accuracy. Production accuracy must be established against a labelled validation set and a physical reference method.

## Replace these backend root files

- `app.py`
- `aslr_engine.py`
- `vision_qa.py`

Optional regression test file:

- `tests/test_v101_36_measurement_reliability.py`

No database migration is required. No frontend file is required for this patch. The existing ASLR QA panel will continue to display the backend overlay.

## Corrections

### 1. ASLR geometry

- Uses the selected raised-leg YOLO hip, knee and ankle chain.
- Measures the selected raised-leg hip-to-ankle line against the original-image horizontal.
- Removes the shared pelvis midpoint from the final angle.
- Keeps the one-call dedicated ASLR pose-model architecture.
- Uses knees and the resting leg only as capture-quality checks.
- Keeps the visual QA payload ephemeral; it is not persisted in the authoritative screening row.

### 2. Shoulder geometry and screening bands

- Uses shoulder-to-elbow humerus orientation as the primary arm vector.
- Uses the wrist only as an elbow-extension check or fallback.
- Selects the most geometrically coherent visible chain when COCO left/right labels are unreliable in a side-view photo.
- Default bands:
  - red: below 160 degrees
  - yellow: 160 to below 175 degrees
  - green: 175 degrees and above
- Therefore a valid 170-degree result is yellow, not green.

The bands are configurable in Render:

- `FLEXILAB_SHOULDER_RED_MAX_DEG=160`
- `FLEXILAB_SHOULDER_GREEN_MIN_DEG=175`

### 3. Posture and squat quality gates

- Rejects insufficiently reliable landmarks instead of returning a confident-looking score.
- Posture selects one coherent ear-shoulder-hip side chain.
- Squat selects one coherent shoulder-hip-knee-ankle side chain instead of averaging incompatible left/right landmarks.
- Existing score policies remain unchanged.

### 4. Retake integrity

- Computes a SHA-256 fingerprint from the uploaded image before job reuse.
- Reuses a completed job only for the identical image.
- A different image in the same session/test is reanalysed and replaces the previous screening result.
- A different image cannot silently receive an older completed score.

### 5. Performance profiling

Each result includes:

```json
"performance_timing_ms": {
  "normalize": 0,
  "pose_inference": 0,
  "geometry_and_quality_gates": 0,
  "vision_qa_render": 0,
  "total_engine": 0
}
```

Render logs also contain `analysis_perf` and `analysis_submit_perf` phase timings.

ASLR input size can be tuned without another code patch:

- Accuracy-first baseline: `FLEXILAB_ASLR_POSE_IMGSZ=960`
- Benchmark candidate after accuracy validation: `FLEXILAB_ASLR_POSE_IMGSZ=832`

Do not lower the value until the same-image repeatability and ground-truth error remain acceptable.

## Automated verification completed

- Python syntax compilation: passed.
- New V101.36 focused regression tests: **9 passed**.
- Full inherited suite after applying the patch to the complete backend: **43 passed, 20 failed**.
- The current unmodified backend baseline was **34 passed, 20 failed**.
- Therefore this patch adds nine passing measurement/reanalysis tests and introduces no additional inherited-suite failure. The 20 existing failures are stale version-contract tests already failing in the supplied backend before this patch.

## Deployment

1. Back up the current three backend files.
2. Upload the replacement files at the repository root.
3. Add the test file under `tests/`.
4. Commit with:

   `Fix measurement geometry and image-aware retakes`

5. Wait for Render deployment.
6. Open `/health` and confirm:

   `V101.36.0-measurement-reliability-and-retake-fingerprint`

7. Keep the following initial Render settings:

   - `FLEXILAB_ASLR_POSE_MODEL=yolo11m-pose.pt`
   - `FLEXILAB_ASLR_POSE_IMGSZ=960`
   - `FLEXILAB_POSE_IMGSZ=640`
   - `FLEXILAB_SHOULDER_RED_MAX_DEG=160`
   - `FLEXILAB_SHOULDER_GREEN_MIN_DEG=175`

## Controlled validation protocol

Do not judge the patch from one mixed screening. Use the same stored photos and a new screening session.

### Shoulder left and right

- Run each reference photo three times.
- The supplied near-170-degree photo must be classified yellow.
- Check `arm_point_used` is `ELBOW_PRIMARY`.
- Check the selected hip, shoulder and elbow form the visible body chain.
- Engineering repeatability target: maximum spread no more than 3 degrees across identical-photo runs.

### ASLR left and right

- Run each supplied photo three times.
- Check the QA overlay labels the hip on the raised leg, not a shared pelvis midpoint.
- Check the selected hip, knee and ankle are on the same raised leg.
- Check the resting leg remains close to horizontal and both knees pass extension gates.
- Engineering repeatability target: maximum spread no more than 3 degrees across identical-photo runs.

### Posture and squat

- Run each stored photo three times.
- Confirm the `side_used` remains stable.
- Confirm low-confidence or cropped landmarks trigger a retake instead of a score.
- Engineering repeatability target: maximum spread no more than 3 degrees across identical-photo runs.

### Retake test

Within the same screening session and same test:

1. Submit photo A.
2. Submit photo A again: the identical capture may be reused.
3. Submit photo B: it must run a new inference and report `force_reanalysis=true`.
4. Confirm the capture-fingerprint prefix changes.

### Performance test

Ignore the first request after a Render cold start. For the next three warm requests, record:

- total user-visible time;
- `pose_inference`;
- `total_engine`;
- image byte size;
- test type.

Only after accuracy is stable should ASLR be benchmarked at image size 832. Restore 960 immediately if angle error or repeatability worsens.

## Scientific/engineering interpretation

- COCO pose models expose 17 body keypoints with per-keypoint coordinates and confidence. Confidence must be combined with geometric coherence; a high confidence score alone does not prove that the correct limb chain was selected.
- Shoulder flexion is an active humeral elevation measurement. A large population study reported average active flexion below 180 degrees, so 180 degrees should not be treated as a universal population norm. The FlexiLab red/yellow/green bands remain product screening references, not diagnostic thresholds.
- Continuous ASLR angle is preferable to over-interpreting coarse categorical FMS ASLR scores; published work found substantial overlap in flexibility across categorical ASLR groups.
- Camera-based joint-angle systems can be repeatable while still showing joint-specific errors, particularly around shoulder flexion. FlexiLab therefore needs a labelled validation dataset rather than relying only on model confidence.

