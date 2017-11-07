# Kera Tutorial - The Happy House

## Objectives

* Learn Kearas: Keras is a high level neural network API writen in Python and able to run on top of several lowel level frameworks like Tensorflow and CNTK
* Learn how to build a Deep Learning Algorithm

## Building the Model

* [Keras Documentation](https://keras.io/models/model/)
* It is really simple to prototype and experiment quickly with Keras
* Workflow
  1. Create Model: see code below
  2. Compile Model: `model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics = ["accuracy"])`
  3. Fit model on training data: `model.fit(x = X_train, y = Y_train, epochs = 50, batch_size = 16)`
  4. Evaluate model performance on test set: `preds = happyModel.evaluate(x = X_test, y = Y_test)`
* Results
  * 99% accuracy training set
  * 95% accuracy test set
  * 50 epochs ran

```python
X_input = Input(input_shape)

# Zero-Padding: pads the border of X_input with zeroes
X = ZeroPadding2D((3, 3))(X_input)

# CONV -> BN -> RELU Block applied to X
X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Dropout(0.2)(X)
X = Activation('relu')(X)

# MAXPOOL
X = MaxPooling2D((2, 2), name='max_pool0')(X)

# CONV -> BN -> RELU Block applied to X
X = Conv2D(64, (5, 5), strides = (1, 1), name = 'conv1')(X)
X = BatchNormalization(axis = 3, name = 'bn1')(X)
X = Dropout(0.2)(X)
X = Activation('relu')(X)

# MAXPOOL
X = MaxPooling2D((2, 2), name='max_pool1')(X)

# CONV -> BN -> RELU Block applied to X
X = Conv2D(128, (5, 5), strides = (1, 1), name = 'conv2')(X)
X = BatchNormalization(axis = 3, name = 'bn2')(X)
X = Dropout(0.2)(X)
X = Activation('relu')(X)

# FLATTEN X (means convert it to a vector) + FULLYCONNECTED
X = Flatten()(X)
X = Dense(1, activation='sigmoid', name='fc')(X)

# Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
model = Model(inputs = X_input, outputs = X, name='HappyModel')
```

## Tips & Tricks

* `model.summary()`: displays the details of your layers
* You can plot your model in a dot graph format
