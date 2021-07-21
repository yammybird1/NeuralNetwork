import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import torch
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1000
epochs = 20

# Choose 1 for the MNIST dataset and 2 for the fashion MNIST dataset
selection = input("Choose which dataset you want to use: ")

if selection == '1':
    batch_size = 1000
    epochs = 20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # print the shape of training input and training targets
    print("shape of x_train:", x_train.shape, "shape of y_train:", y_train.shape)

    # there are 60,000 training data of image size 28x28 and 60,000 train targets
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

    # Print the number of training and test datasets
    print(x_train.shape[0], 'train set')
    print(x_test.shape[0], 'test set')

    # Normalize the data dimensions so they are approximately the same scale
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Print training set shape
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

    # Create the neural network
    model = tf.keras.Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='sigmoid', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=48, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(Conv2D(filters=48, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=96, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(Conv2D(filters=96, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(Conv2D(filters=96, kernel_size=3, padding='same', activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    opt = keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.15, verbose=1)

    # list all data in history
    print(history.history.keys())

    # plot the training accuracies and test losses
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    loss_epochs = range(1, 21)
    plt.plot(loss_epochs, loss_train, 'g', label='Training loss')
    plt.plot(loss_epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot the training accuracies and test accuracies
    acc_train = history.history['accuracy']
    acc_val = history.history['val_accuracy']
    acc_epochs = range(1, 21)
    plt.plot(acc_epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(acc_epochs, acc_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

elif selection == '2':
    batch_size = 1000
    epochs = 20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # print the shape of training input and training targets
    print("shape of x_train:", x_train.shape, "shape of y_train:", y_train.shape)

    # there are 60,000 training data of image size 28x28 and 60,000 train targets
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

    # Print the number of training and test datasets
    print(x_train.shape[0], 'train set')
    print(x_test.shape[0], 'test set')

    # Normalize the data dimensions so they are approximately the same scale
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


    # Print training set shape
    print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

    # Create the neural network
    model = tf.keras.Sequential()

    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='elu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=48, kernel_size=3, padding='same', activation='elu'))
    model.add(Conv2D(filters=48, kernel_size=3, padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='elu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=96, kernel_size=3, padding='same', activation='elu'))
    model.add(Conv2D(filters=96, kernel_size=3, padding='same', activation='elu'))
    model.add(Conv2D(filters=96, kernel_size=3, padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(512, activation='elu'))
    model.add(Dense(512, activation='elu'))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    opt = keras.optimizers.Adam(learning_rate=0.002)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.15, verbose=1)

    # list all data in history
    print(history.history.keys())

    # plot the training losses and test losses
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    loss_epochs = range(1, 21)
    plt.plot(loss_epochs, loss_train, 'g', label='Training loss')
    plt.plot(loss_epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot the training accuracies and test accuracies
    acc_train = history.history['accuracy']
    acc_val = history.history['val_accuracy']
    acc_epochs = range(1, 21)
    plt.plot(acc_epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(acc_epochs, acc_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
