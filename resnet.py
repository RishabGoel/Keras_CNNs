import keras
from keras.layers import Dense, Flatten, Activation, Input, Concatenate
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU,Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
import numpy as np

class AccuracyHistory(keras.callbacks.Callback):
    """
    This class accumulates accuracy over epochs
    """
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def make_model(input_shape = (32,32,3)):
    # Declare the input layer. The input shape should be batch_size,32,32,3
    inputs = Input(input_shape)

    # Next line performs convolution of kernel of receptive field 3x3xdepth_of_input over the inputs and genrate 64 feature maps
    x = Conv2D(64, kernel_size = (3,3), padding='same')(inputs)
    # Batchnormalization is performed to improve the statistics of the output that improves the training stability
    x = BatchNormalization()(x)
    # Leaky relu is applied to obtain the final 64 feature maps
    x = LeakyReLU()(x)
    # The next line is the skip connection from input to this point. This skip connection is the actual strength 
    # of resnets as this helps in better backward flow of gradients thus reducing vanishing gradient problem. It also
    # helps to combine fine grained and coarse grained features.
    x = Concatenate()([inputs, x])
    # To obtain a more robust representation and reduce dimensionality, the next line performs maxpooling on a window od
    # size 2x2
    x1 = MaxPooling2D((2,2))(x)
    # Droput is a regularization method that prevent overfitting and helps in better generalisation. Adding the following 
    # dropout layer zzeros out a neuron with a probability of 25%.
    x = Dropout(0.25)(x1)
    # Next line performs convolution of kernel of receptive field 3x3xdepth_of_input over the inputs and genrate 128 feature maps
    x = Conv2D(128, kernel_size = (3,3), padding='same')(x)
    # Batchnormalization is performed to improve the statistics of the output that improves the training stability
    x = BatchNormalization()(x)
    # Leaky relu is applied to obtain the final 64 feature maps
    x = LeakyReLU()(x)
    x = Concatenate()([x1, x])
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)
    # Now that we have got distributed good features, we convert the 3d matrix to a single layer to be passed through
    # a fully connected layer
    x = Flatten()(x)
    # Dense adds a fully connected layer  of 128 units for allowing more composition of features.
    c = Dense(128)(x)
    c = BatchNormalization()(c)
    c = LeakyReLU()(c)
    c = Dropout(0.25)(c)
    # Out problem is binary classification we add a new dense layer for final predicition of the class
    c = Dense(1)(c)
    c = BatchNormalization()(c)
    # Sigmoid direcly gives the probability if a class is present or not
    c = Activation('sigmoid')(c)
    # we convert the above operations in model
    model = Model(inputs,  c)
    # This setsup the loss function and optimizer that we want to use. We can also add anothe metrics like accuracy or precisionet as well
    model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy',metrics=['accuracy'])
    return model

batch_size = 20
epochs = 100
img_x, img_y = 28, 28


(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_idc = np.concatenate((np.where(y_train==0)[0] , np.where(y_train==1)[0]))
x_train,y_train = x_train[train_idc], y_train[train_idc]
test_idc = np.concatenate((np.where(y_test==0)[0] , np.where(y_test==1)[0]))
x_test,y_test = x_test[test_idc], y_test[test_idc]

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
input_shape = (img_x, img_y, 1)

# performing normalization to avoid unstable gradient jumps around the minima
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = make_model((28,28,1))

history = AccuracyHistory()
# Training and get validationaccuracy as well
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(1, 11), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

