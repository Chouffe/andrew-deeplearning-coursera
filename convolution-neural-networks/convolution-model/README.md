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

### Forward Propagation

* Tensorflow takes care of carrying out the convolution steps
  * `tf.nn.conv2d(X, W1, strides = [1,s,s,1], padding = 'SAME')`
    * It convolves `X` with kernels `W1`s
    * [TF Doc here](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
    * Input `X`
    * Kernels: `W1`
    * Strides: `strides = [1, s, s, 1]`
    * Padding `SAME` or `VALID`
  * `tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME')`
    * It performs max pooling
    * [TF Doc here](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)
  * `tf.nn.relu(Z)`
    * It performs the elementwise ReLU of `Z`
    * [TF Doc here](https://www.tensorflow.org/api_docs/python/tf/nn/relu)
  * `tf.contrib.layers.flatten(P)`
    * It flattens into a 1D vector while maintaining the batch-size -> `[batch_size, k]`
    * [TF Doc here](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten)
  * `tf.contrib.layers.fully_connected(F, num_outputs)`
    * It returns the output using a FC
    * `F`: flattened input
    * [TF Doc here](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected)
    * It automatically intializes the weights of the FC layer

### Compute Cost

* `tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)`
  * It computes the softmax entropy loss
    * Softmax activation function
    * Resulting loss
    * [TF Doc here](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)
* `tf.reduce_mean`
  * Computes the meam of elments across dimensions of a tensor
  * It is used to sum the losses over the batch examples to get the overall cost
  * [TF Doc here](https://www.tensorflow.org/api_docs/python/tf/reduce_mean)

### Model

* Do not forget to initialize global variables: `init = tf.global_variables_initializer()`
* Run the initialization: `sess.run(init)`
* AdamOptimizer minimizing the cost: `optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)`
* Run the computation graph on a minibatch: `_, temp_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y}`
