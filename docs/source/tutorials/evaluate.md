# Scoring attributions using QuAC

At this point, we will have generated a set of candidate regions, using various heuristic methods. 
Now, we will obtain the mask, counterfactual, and score for our final explanation using the `run_evaluation.py` script.

It has the following arguments, all of which are optional:
- `dataset`: Which of the datasets to run the translation on. By default this will be the "test" dataset, if that does not exist it will revert to the "validation" dataset.
- `attrs`: Where the attributions are. You should only set this if you used a custom `output` argument in the script above. By default, this is in the experiment root directory under `attributions`
- `input_fake`: Where the generated images are. You should only set this if you used a custom `output` argument in the image generation step. By default, this is in the experiment root directory under `generated_images`
- `names`: A selection of attribution methods to run evaluation on. This is a useful argument if you want to run evaluation on each method simultaneously, *e.g.* on a cluster. By default, we will sequentially run evaluation on all the methods in the `attrs` directory.

To run using the defaults, simply run: 
```{code-block} bash
python run_evaluation.py
```

You can use the following command to get help with formatting your arguments.
```{code-block} bash
python run_evaluation.py -h
```


## Output

Here is the output organization that you should expect at this point.
```{code-block} bash
<solver.root_dir>/
├── checkpoints/
├── generated_images/
├── attributions/
├── counterfactuals/
    ├── discriminative_deeplift/
    │   └── class_A/class_B/... # image files
    └── discriminative_ig/
    │   └── class_A/class_B/...
├── masks/
    ├── discriminative_deeplift/
    │   └── class_A/class_B/... # numpy files
    └── discriminative_ig/
    │   └── class_A/class_B/...
└── reports/
    ├── discriminative_deeplift/
    │   └── default.json
    └── discriminative_ig/
        └── default.json
```