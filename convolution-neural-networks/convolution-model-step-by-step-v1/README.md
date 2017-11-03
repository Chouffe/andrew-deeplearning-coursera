# Convolutional Neural Networks: Step by Step

## Goal

Implementation of CONV and POOL layers in numpy, including forward and backward propagation

## Zero padding

```python
import numpy as np

# Example of zero 2-padding with numpy
np.pad(np.ones((10, 10)), ((2, 2), (2, 2)), 'constant', constant_values = (0, 0))
np.pad(np.ones((100, 10, 10)), ((0, 0), (2, 2), (2, 2)), 'constant', constant_values = (0, 0))
```

## Max/Average POOLING

* `np.max` and `np.mean` are useful to compute the max and mean on a slice
* Backpropagation
  * **Max Pooling**
    * Implement a mask where the max is
    * The max value is the input value that ultimately influenced the output, and therefore the cost
    * Anything that influenced the cost should have a non-zero gradient
  * **Average Pooling**
    * Implement a mask with `dZ / size mask` as values
    * Each position in the `dZ` matrix contributes equally to the output because we took average in the forward pass
