# Train the generator

A source-only YAML file is sufficient:

```yaml
data:
  source: images
```

```sh
quac train --config config.yaml
```

StarGAN is the default backend. Class count, channels, and image dimensions are
inferred. Defaults include 100000 steps, batch size 4, learning rate 0.0001,
mapping-network learning rate 0.000001, and EMA beta 0.999. Override values under
`training` and architecture values under `generator`. Use
`quac.config.Config.model_json_schema()` for all fields and defaults.

The final checkpoint is `quac-output/checkpoints/latest.pt`. To resume, set
`generator.checkpoint` to a native checkpoint and increase `training.steps` to the
desired final step. Configurations reject unknown keys.
