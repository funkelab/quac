# Pair attribution

Select attribution methods in the same YAML file:

```yaml
explanation:
  methods: [DIntegratedGradients, residual]
```

QuAC includes discriminative integrated gradients, DeepLift, input gradients,
classical integrated gradients and DeepLift baselines, residuals, and random maps.
Discriminative methods use the generated image as their comparison input.
Importance is converted to a finite `[0,1]` spatial map; constant maps become zero.
DeepLift needs an eager classifier supporting Captum hooks; use Python injection
for that method.

Custom Python attributions implement `attribute(real, generated, source, target)`
and are supplied as a name-to-component dictionary to `explain`. Names must match
`explanation.methods`.
