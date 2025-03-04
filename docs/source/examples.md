# Example gallery

Here are some examples of datasets that have been investigated with QuAC, as well as what was found.
If you have tried QuAC on your own dataset (even unsuccessfully!) please [contact us](mailto:adjavond%40hhmi.org?subject=QuAC%20Example) so we can add it to the gallery.


## *Fictus Aggregatum* synthetic cells

The *Fictus aggregatum* dataset is a synthetic dataset that was created in the [Funke Lab](https://www.janelia.org/lab/funke-lab) specifically to understand how QuAC works on biological data.
The code to generate this dataset is available [here](https://github.com/funkelab/fictus.aggregatum).

```{figure} assets/fictus.png
    :figwidth: 100%
    :alt: Fictus aggregatum
    :align: center

An example of the query (top) and counterfactual (bottom) images, highlighting the differences.
```

We evaluated, using this *fictus* dataset, whether QuAC was able to retrieve all of the differences between classes when these are known.
We found that, although this was sometimes done in surprising ways, the changes described by QuAC were generally in line with what was expected from the data.


## *Drosophila melanogaster* synapses

The differences between synapses emitting different neurotransmitters in the fruit fly *Drosophila melanogaster* are so subtle that it was not though possible to tell them apart.
When [it was found that a deep learning model could do so](https://www.cell.com/cell/fulltext/S0092-8674(24)00307-6), however, this opened up possibilities for gaining insight into the relation between strcuture and function in these synapses.

```{figure} assets/synapses.png
    :figwidth: 100%
    :alt: Synapses in EM
    :align: center

A few examples of synapses in the *Drosophila* brain, as seen in electron microscopy, translated from one class to another.
```

QuAC explanations suggested quite a few new features that could be used to distinguish between these synapse types, the prevalence of which is currently being investigated.


## Fly identity


```{attention}
    This dataset is still under construction. Come back soon for updates!
```