# Convolutional Neural Networks

## Step by Step

### Goal

Implementation of CONV and POOL layers in numpy, including forward and backward propagation

### Zero padding

```python
import numpy as np

# Example of zero 2-padding with numpy
np.pad(np.ones((10, 10)), ((2, 2), (2, 2)), 'constant', constant_values = (0, 0))
np.pad(np.ones((100, 10, 10)), ((0, 0), (2, 2), (2, 2)), 'constant', constant_values = (0, 0))
```

### Max/Average POOLING

* `np.max` and `np.mean` are useful to compute the max and mean on a slice
* Backpropagation
  * **Max Pooling**
    * Implement a mask where the max is
    * The max value is the input value that ultimately influenced the output, and therefore the cost
    * Anything that influenced the cost should have a non-zero gradient
  * **Average Pooling**
    * Implement a mask with `dZ / size mask` as values
    * Each position in the `dZ` matrix contributes equally to the output because we took average in the forward pass

## Application

### Goal

Implement a fully functioning ConvNet using TensorFlow

### Tensorflow placeholders

* Resource: ![Doc Placeholders](https://www.tensorflow.org/api_docs/python/tf/placeholder)
```python
import tensorflow as tf

X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
Y = tf.placeholder(tf.float32, shape=(None, n_y))
```

### Intialization

* Use Xavier Intialization: `tf.contrib.layers.xavier_initializer(seed = 0)`
* TensorFlow handles the biases

```python
# Example: Initialize a parameter W of shape [1,2,3,4] in tf
W = tf.get_variable("W", [1,2,3,4], initializer = ...)

W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
```
