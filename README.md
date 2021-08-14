# watershed-segmentation

A simple and optimised Python implementation of a type of Image Segmentation called the Watershed Segmentation.

The watershed is a classical algorithm used for segmentation, that is, for separating different objects in an image.Starting from user-defined markers, the watershed algorithm treats pixels values as a local topography (elevation)[[1]](#1). 

We are given an input image and the corresponding seed points for that image. Seed points are just pixel locations of a few points in each expected region of the segmentation. We start from this list of seeds and grow our regions.

The end results of this model look like the image below, where the image on the left is the input and the image on the right is the output.
![alt text](https://github.com/a-manic/watershed-segmentation/blob/main/outputs/Result.jpg)

## References
<a id="1">[1]</a>
https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
