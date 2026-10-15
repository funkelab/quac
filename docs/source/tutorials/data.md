# Image data

Put images in one subdirectory per class. Classes are ordered alphabetically.
Images must share their dimensions and grayscale/RGB channel count; these are
inferred automatically. Floating-point images must be finite in `[0,1]`.
Integer images are scaled by their dtype maximum. There is no per-image min/max
normalization. Float TIFF preserves full precision.

```yaml
data:
  source: images
```

Paths in YAML are relative to the config file. Generator-specific padding and
signed scaling happen inside the StarGAN backend.
