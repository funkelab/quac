# Refactor implementation notes

## 1. Core components and StarGAN backend

- Added strict monolithic configuration with inferred dataset metadata.
- Established float32 CHW images in [0,1], with signed scaling and spatial padding confined to StarGAN adapters.
- Separated classifier inference, pair attribution, StarGAN losses, optimization, EMA, and checkpoints.
- Preserved compatible legacy EMA weight loading; native checkpoints also store optimizer and RNG state for resume.
- Validation: 27 focused tests passed, including grayscale/RGB I/O, attributions, numerical evaluation, real StarGAN training, checkpoint reload, and resume.
