import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import keras.models as km
import keras.layers as kl
import keras.utils as ku
import keras.optimizers as ko
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# setup CNN model


def cnn_model(numfm, numnodes, input_shape=(28, 28, 1), output_size=10):

    # Initialize the model.
    model = km.Sequential()

    # Add a 2D convolution layer, with numfm feature maps.
    model.add(kl.Conv2D(numfm, kernel_size=(5, 5),
                        input_shape=input_shape,
                        activation='relu'))
    model.add(kl.Conv2D(numfm, kernel_size=(5, 5),
                        activation='relu'))
    # Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Dropout(0.25))

    # Second layer
    model.add(kl.Conv2D(numfm * 2, kernel_size=(3, 3),
                        activation='relu'))
    model.add(kl.Conv2D(numfm * 2, kernel_size=(3, 3),
                        activation='relu'))
    # Add a max pooling layer.
    model.add(kl.MaxPooling2D(pool_size=(2, 2),
                              strides=(2, 2)))
    model.add(kl.Dropout(0.25))

    # Convert the network from 2D to 1D.
    model.add(kl.Flatten())

    # Add a fully-connected layer.
    model.add(kl.Dense(numnodes,
                       activation='relu'))
    model.add(kl.Dropout(0.5))
    # Add the output layer.
    model.add(kl.Dense(10, activation='softmax'))

    # Return the model.
    return model


# Load the data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

# separate x_train and y_tain
x_train = train.drop(labels=['label'], axis=1)

# Normalize the data
x_train = x_train / 255.0
test = test / 255.0

y_train = train['label']
y_train = ku.to_categorical(y_train, 10)
del train

# reshape the data
x_train = x_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# Set the random seed
random_seed = 2
# Split the train and the validation set for the fitting
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=random_seed)

x_train.shape

# employ model
model = cnn_model(16, 256)
model.summary()
# Define the optimizer
optimizer = ko.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',  patience=3,  verbose=1,
                                            factor=0.5,  min_lr=0.00001)

datagen = ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=20,
                             zoom_range=0.2,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=False,
                             vertical_flip=False)


datagen.fit(x_train)

batch_size = 100
fit = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                          epochs=30, validation_data=(x_val, y_val), steps_per_epoch=2*x_train.shape[0] // batch_size,
                          verbose=2, callbacks=[learning_rate_reduction])
# predict results
results = model.predict(test)
# select the indix with the maximum probability
results = np.argmax(results, axis=1)
results = pd.Series(results, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("digit_recognizer_cnn.csv", index=False)
