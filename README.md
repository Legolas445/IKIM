# Translational Image-Guided Oncology: Coding Challenge

Welcome to the coding challenge of the research group Translational Image-guided Oncology at the newly founded Institute for Aritficial Intelligence in Medicine (IKIM) at the University Hospital Essen.

We are excited that you are applying to be part of the research group and the institute. If you've made it this far in the selection process, there must be something unique and impressive about you.

The aim of the coding challenge is to get a sense of your _approach to solving problems_, your _technical coding abilities_, and your _grasp of machine learning methods_. 

There is __no single correct answer__ to the challenge. Instead, we encourage you to play to your technical strengths, to experiment, to visualize your results and to document your work, so we can get an idea how you approach problems.

## The Task

The goal of the coding challenge is for you to __develop a basic machine learning model__ that takes as inputs two `.nrrd` files (`.nrrd` is a common file format in medical image computation, see [some pointers](#some-pointers)) and generates as output a single label.

- The first _input_ NRRD-file is a 15x15 pixel area of an organ, isolated from an MRI sequence. 
- The second _input_ NRRD-file is a 15x15 binary mask, specifying the region of interest (ROI) in the first file.
- The _output_ should be the name of the organ to which the first file belonged.

Your final submission should contain all the code you used to train your model, your final model, any code you used to generate visualizations. Moreover, please include an executable script/ function that loads your final model and makes predictions and reports accuracy for any two directories containing, respectively, the first and second input `.nrrd` files, as described above. 

In total, we intend for you to spend __no more than about 4 hours__ on the coding challenge.

Some other considerations:

- Start by cloning this repository. 
- When you are done, push the code and the results of your work. 
- The coding challenge should be completed using Python 3. You can use Jupyter Notebooks or standard Python scripts.
- We encourage the use of common machine learning packages, e.g., [scikit-learn](https://scikit-learn.org/stable/), [keras](https://keras.io/) or [pytorch](https://pytorch.org/). 
- Please include a `requirements.txt` in your submission, so that we can reproduce your code.
- Should you want to make use of a GPU-based machine learning method, you are encouraged to do so. For instance, Google [Colab](https://pytorch.org/) offers free GPU resources.
 

## The Data

The data is part of this repository and can be found in `./data`. It is organized as follows:

```
data
+--- images
|        1_liver7.nrrd
|        ...
+--- masks
|        1_liver7.nrrd
|        ...
```

Associated images (first input file described above) and masks (second input file described above) are labeled identically. So, `./data/masks/1_liver7.nrrd` is the ROI for `./data/images/1_liver7.nrrd`. 

The first number in the file name,  so `1` in `1_liver7.nrrd`, is the site at which the MRI image patch was recorded. In total, there are 7 sites. You have access to the data of 6 sites. We will be evaluating the performance of your model on the data of site 4, which we held back.

The file name also contains the label of the target organ, i.e., the intended output of your model. In this case the target label is `liver`. In total there are 5 different labels, i.e classes: liver, kidney, spleen, muscle and bone.


## Some Pointers

- To load `.nrrd` files you can use [pynrrd](https://pypi.org/project/pynrrd/). It provides easy access to the data in the form of a numpy array. 
- One possible approach to the problem is to start by extracting radiomics features from the images using [pyradiomics](https://pyradiomics.readthedocs.io/en/latest/) and then to use some of those features to train a model.
- An easy way to visualize images and masks is with [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php).

## Acknowledgements
This challenge was prepared by `jacob.murray@uk-essen.de`

