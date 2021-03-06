import keras
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU,Dropout
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
import numpy as np

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

def make_model(input_shape = (32,32,3)):
	# initialising a sequential cnn
	model = Sequential()
	# Next line performs convolution of kernel of receptive field 3x3xdepth_of_input over the inputs and genrate 64 feature maps
	model.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', input_shape=input_shape))
	# Batchnormalization is performed to improve the statistics of the output that improves the training stability
	model.add(BatchNormalization())
	# Leaky relu is applied to obtain the final 64 feature maps
	model.add(LeakyReLU())
	# To obtain a more robust representation and reduce dimensionality, the next line performs maxpooling on a window od
    # size 2x2
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# Droput is a regularization method that prevent overfitting and helps in better generalisation. Adding the following 
    # dropout layer zzeros out a neuron with a probability of 25%.
	model.add(Dropout(0.25))
	model.add(Conv2D(128, kernel_size=(3, 3), padding = 'same'))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(Dropout(0.25))
	# Now that we have got distributed good features, we convert the 3d matrix to a single layer to be passed through
    # a fully connected layer
	model.add(Flatten())
	# Dense adds a fully connected layer  of 128 units for allowing more composition of features.
	model.add(Dense(128))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(Dropout(0.25))
	model.add(Dense(1))
	model.add(BatchNormalization())
	# Sigmoid direcly gives the probability if a class is present or not
	model.add(Activation('sigmoid'))
	# This setsup the loss function and optimizer that we want to use. We can also add anothe metrics like accuracy or precisionet as well
	model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy',metrics=['accuracy'])
	return model



batch_size = 20
epochs = 100
img_x, img_y = 28, 28
# num_classes = 1

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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = make_model(input_shape)

history = AccuracyHistory()

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

