# Deep Learning & Art: Neural Style Transfer

* In Neural Style Transfer, a cost function is optimized to get pixel values
* The Neural Network parameters are fixed

## Transfer Learning

* NST uses a pretrained ConvNet
* ImageNet pretrained VGG-19 was used in the [original paper](https://arxiv.org/pdf/1508.06576.pdf)
* Run an image through the network: `tf.assign(image)`
* Retrieve activations of a particular layer: `sess.run(model["conv4_2"])`

## Neural Style Transfer

### Content Cost

$$J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2\tag{1} $$

### Style Cost

$$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{ij} - G^{(G)}_{ij})^2\tag{2} $$
