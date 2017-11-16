# Car Detection for Autonomous Driving

## YOLO

* [Official website](https://pjreddie.com/darknet/yolo/)

### Filtering with a threshold on class scores

* find classes: `K.argmax(box_scores, axis=-1)`
* find class scores: `K.max(box_scores, axis=-1)`
* Applying boolean mask: `tf.boolean_mask(tensor, mask)`

### Non Max Suppression

* Tensorflow
  * `tf.image.non_max_suppression()`
  * Greedily selects a subset of bounding boxes in descending order of score
  * [Doc](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression)
* Keras - tensorflow backend
  * [Doc](https://www.tensorflow.org/api_docs/python/tf/gather)
  * Gather slices from `params` axis `axis` according to `indices`

### Prediction

* When using BatchNorm, tf `feed_dict` param has an extra parameter: `{K.learning_phase(): 0}`
