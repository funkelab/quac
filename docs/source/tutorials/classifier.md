# Classifier contract

Provide a TorchScript classifier accepting float32 `[N,C,H,W]` images in `[0,1]`
and returning finite `[N,num_classes]` logits. Output columns follow sorted class
directory names. QuAC stores logits and softmax probabilities without changing
classifier inputs.

```yaml
classifier:
  checkpoint: classifier.pt
```

Classifier configuration is required for `explain`, not `train`. The example
setup script exports a matching classifier. Python callers may inject an eager
`torch.nn.Module` through `explain(config, classifier=model)`.
