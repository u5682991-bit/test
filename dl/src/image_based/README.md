# Image-Based Literature Baselines

This folder is reserved for image-based methods such as MAP-Net.

Do not register MAP-Net in `src/models/factory.py`, because that factory is for
patch-based matching models:

```text
G2-G7: single-scale patch matching
G9: multi-scale patch matching and fusion
```

MAP-Net should stay as a separate full-image pipeline:

```text
full SAR + full Optical
dense feature extraction
SPAP / attention
PCA / KD-tree matching
RANSAC
affine or homography registration
```

