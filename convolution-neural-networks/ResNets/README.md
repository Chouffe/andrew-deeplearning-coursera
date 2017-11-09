# Residual Networks

## Objectives

* Implement the basic building blocks of ResNets
* Put together these building blocks to implement and train a state-of-the-art neural network for image classification

## Problem of very deep Neural Networks

* In recent years, NNs have become deeper - from `AlexNet` (a few layers) to `Inception Networks` (> 100 layers)
* **Pros**
  * It can represent very complex functions
  * It can also learn features at many different levels of abstraction
    * from edges (at the lower layers)
    * to very complex features (at the deeper layers)
* **Cons**
  * Vanishing / Exploding gradients
  * Training error increases as depth increases
* There are 2 types of residual blocks
  * Identity Block
  * Convolutional Block

## The Identity Bloxk

* `a[l]` and `a[l + 2]` have the same dimensions
* One can skip over 2 or 3 layers

## The Convolutional block

* `a[l]` and `a[l + 2]` do not have the same dimensions
* The CONV layer in the shortcut path is used to resize the input `X` to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path
* Its main role is to just apply a (learned) linear function that reduces the dimension of the input, so that the dimensions match up for the later addition step.

## ResNet-50

* When adding the skip connections use Keras `Add()` class instead of `+`: `X = Add()([X, X_shortcut])`
* ResNet-50 is State of the Art image classification

## Conclusions

* Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients
* The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function.
* There are two main type of blocks: The identity block and the convolutional block.
* Very deep Residual Networks are built by stacking these blocks together.

## Resources

* [ResNet Open Source implementation](https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py)
* [ResNet Paper - Deep Residual Learning for Image Recognition - 2015](https://arxiv.org/pdf/1512.03385.pdf)
