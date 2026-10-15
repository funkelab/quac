# Pair attribution

Select attribution methods in the same YAML file:

```yaml
explanation:
  methods: [DIntegratedGradients, residual]
```

QuAC includes discriminative integrated gradients, DeepLift, input gradients,
classical integrated gradients and DeepLift baselines, residuals, and random attribution maps.
Discriminative methods use the generated image as their comparison input.
Attribution maps are normalized to finite `[0,1]` spatial arrays; constant
attribution maps become zero.
DeepLift needs an eager classifier supporting Captum hooks; use Python injection
for that method.

Custom Python attributions implement `attribute(source_image, generated_image, source_class, target_class)`
and are supplied as a name-to-component dictionary to `explain`. Names must match
`explanation.methods`.
