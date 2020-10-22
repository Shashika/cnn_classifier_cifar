from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import MaxPooling2D, Multiply, Conv2D
from keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.datasets import cifar10

class CNNClassifier:

    def __init__(self):
        self.batch_size = 64
        self.epochs = 2
        self.num_classes = 10
        self.dict = None
        self.trained_model = None

        self.train_X = None
        self.train_Y = None
        self.train_label = None
        self.test_X = None
        self.test_Y = None
        self.test_label = None
        self.predicted_Y = None
        self.steps = 0
        self.it_train = None

        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        #choose relavent model (build_cnn / build_cnn_with_batch_norm)
        self.cnn = self.build_cnn_with_batch_norm()

        # compile model
        # opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        # opt = RMSprop(lr=0.0001, decay=1e-6)
        opt = SGD(lr=0.001, momentum=0.9)
        self.cnn.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def read_data(self):

        # load dataset
        (self.train_X, self.train_Y), (self.test_X, self.test_Y) = cifar10.load_data()
        # one hot encode target values
        self.train_label = to_categorical(self.train_Y)
        self.test_label = to_categorical(self.test_Y)

        self.train_X = self.train_X.astype('float32')
        self.train_X = self.train_X / 255.

        self.test_X = self.test_X.astype('float32')
        self.test_X = self.test_X / 255.

    def count_labels(self, data):
        d = dict()
        for i in range(len(data)):
            if data[i] in d:
                v = d[data[i]]
                d[data[i]] = v + 1
            else:
                d[data[i]] = 1

    def build_cnn(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer="he_normal", padding='same', input_shape=(32, 32, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer="he_normal", padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer="he_normal", padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer="he_normal", padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.summary()
        return model

    def build_cnn_with_batch_norm(self):
        model = Sequential()
        model.add(
            Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                   input_shape=(32, 32, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        return model

    def train(self):
        # create data generator
        datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                               samplewise_center=False,  # set each sample mean to 0
                                               featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                               samplewise_std_normalization=False,  # divide each input by its std
                                               zca_whitening=False,  # apply ZCA whitening
                                               zca_epsilon=1e-06,  # epsilon for ZCA whitening
                                               rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                                               width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
                                               height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
                                               shear_range=0.,  # set range for random shear
                                               zoom_range=0.,  # set range for random zoom
                                               channel_shift_range=0.,  # set range for random channel shifts
                                               fill_mode='nearest', # set mode for filling points outside the input boundaries
                                               cval=0.,  # value used for fill_mode = "constant"
                                               horizontal_flip=True,  # randomly flip images
                                               vertical_flip=False,  # randomly flip images
                                               rescale=None,   # set rescaling factor (applied before any other transformation)
                                               preprocessing_function=None,    # set function that will be applied on each input
                                               data_format=None,    # image data format, either "channels_first" or "channels_last"
                                               validation_split=0.0  # fraction of images reserved for validation (strictly between 0 and 1)
                                               )

        # prepare iterator
        self.it_train = datagen.flow(self.train_X, self.train_label, batch_size=64)
        # fit model
        self.steps = int(self.train_X.shape[0] / 64)
        self.trained_model = self.cnn.fit_generator(self.it_train, steps_per_epoch=self.steps, epochs=self.epochs,
                                                    validation_data=(self.test_X, self.test_label), verbose=2)

    def evaluate(self):
        test_eval = self.cnn.evaluate(self.test_X, self.test_label, verbose=0)
        self.predicted_Y = self.cnn.predict_classes(self.test_X, batch_size=self.batch_size, verbose=2)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

    def draw(self):
        accuracy = self.trained_model.history['acc']
        val_accuracy = self.trained_model.history['val_acc']
        loss = self.trained_model.history['loss']
        val_loss = self.trained_model.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, color='blue', label='Training accuracy')
        plt.plot(epochs, val_accuracy, color='orange', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig('acc.png')

        plt.figure()
        plt.plot(epochs, loss, color='blue', label='Training loss')
        plt.plot(epochs, val_loss, color='orange', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig('loss.png')

        cm = confusion_matrix(self.test_Y, self.predicted_Y)
        plot_confusion_matrix(conf_mat=cm, colorbar=True, show_normed=True)
        plt.savefig('test.png')