# Image classification
The process of labeling the images into finite classes.

## Nearest neighbor classifier
It is done with measuring distances between test image and training images and finding the most smallest distance (i.e. the smallest error rate) pair of images. We can think that the training image of that pair is the most closest image for our test image.

## K-nearest neighbor classifier
Instead of finding the single closest image, we can find the top **k** closest images, and vote on the label of the test image.
Intuitively, higher values of **k** have a smoothing effect that makes the classifier more resistant to outliers

## Hyperparameter
We can choose many different distance functions for classification. Also, we can use lots of classifiers and their setting. e.g. k-nn classifier requres a setting for *k*. These choices are called **hyperparameters**.

### Tuning the hyperparameters
"We cannot use the test set for the purpose of tweaking hyperparameter". It is obvious because the *test set* refer to the set of data for *testing*, not for *training*.
In real world, the model we've designed will be used for data we do not know at the time of training. If we tuned the hyperparameters to fit our test set, we would say that we **overfit** to the test set.

### Validation set
What is the correct way of tuning the hyperparameters without touching the test set? The ideal answer is to split our training set in two: a slightly smaller training set, and what we call a **validation set**.