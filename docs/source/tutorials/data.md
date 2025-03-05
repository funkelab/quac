# Data preparation

## Supported data formats

We support square, 2D images. 
They can be either grayscale or RGB. 
The images must have one of the following file extensions: `["png", "jpg", "jpeg", "JPG", "tiff", "tif"]`. 
All of your images should be of the same type.

## Data splits
You must split your data into a `train` and a `validation` set with different images.
The `validation` dataset is used to determine the best models to use.
We also recommend a separate `test` dataset, which can be used to compute final metrics.
Make sure that there is no overlap between the images in these three datasets, otherwise this may invalidate your results.

We recommend you to have one root directory for all of your data, and sub-directories for `train`,`validation` and `test`. 

```{code-block} bash

root/
    train/
    validation/
    test/
```

## File organization
Within the `train`, `validation` and `test` directories, the sub-directories define class membership.
The images must be organized in directories named after the class they belong to. 
Only the first level of the directory structure is considered as the class. 

Here is an example of a valid organization of the `train` directory in a case with two classes `class_A` and `class_B`.
```{code-block} bash

train/class_A/xxx.png
train/class_A/xxy.png
train/class_A/[...]/xxz.png

train/class_B/123.png
train/class_B/nsdf3.png
train/class_B/[...]/asd932_.png
```

Every image in the `class_A` directory will get the label for `class_A`, including if they are in sub-directories. 

```{attention}
Make sure to use the same names for your class directories in the `train`, `validation`, and `test` directories. 
The names are used to determine the labels, and you will get incorrect results if they do not match, or if you have any additional class directories.
```

## Example data

If you just want to try QuAC as a learning experience, you can use one of the datasets [in this collection](https://doi.org/10.25378/janelia.c.7620737.v1), and the pre-trained classifiers we provide.