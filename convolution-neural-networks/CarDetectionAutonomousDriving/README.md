# Car Detection for Autonomous Driving

## YOLO

### Filtering with a threshold on class scores

* find classes: `K.argmax(box_scores, axis=-1)`
* find class scores: `K.max(box_scores, axis=-1)`
* Applying boolean mask: `tf.boolean_mask(tensor, mask)
