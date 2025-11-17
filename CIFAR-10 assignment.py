# importing standard libraries for AI stuffs (and other stuff prob too)
import seaborn as sns #stats analysis
import tensorflow as tf #generic version of tensorflow, CPU version only    #created/owned by google, idk what exactly its for
import numpy as np #fancy stuff? w/ numbers and python prob
import pandas as pd #dataframe
import matplotlib.pyplot as plt #plot graphs
import matplotlib.image as mpimg #output images
import random #not really required

# print("Tensorflow: ", tf.__version__, "Seaborne: ", sns.__version__)

# define labels 
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# making a dataframe using the cifar-10 dataset
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x = images(32x32 pixels), y = labels(str: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

# # shows how many of each number
# sns.countplot(x = y_train) # x axis of graph = labels 
# plt.show()

# check to make sure that there are NO values that aren't numbers (NaN = not a number)
print("Any NaN Training: ", np.isnan(x_train).any())
print("Any NaN Testing: ", np.isnan(x_test).any())

# tell the model what shape to expect
input_shape = (32, 32, 3) # 32x32 pixels(p0x), 1 color channel for grayscale (3 color channels for RGB)

# reshape the data to include all the images at once
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3) # (50_000, 32, 32, 3) --> doesn't work w/ CiFar10
x_train = x_train.astype('float32') / 255.0 #normalize the data to be between 0 and 1 (relative% instead of 0-255)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
x_test = x_test.astype('float32') / 255.0

# convert labels to be one-hot, not sparse
from keras.utils import to_categorical

y_train = to_categorical(y_train, 10) #labels a category (ex:bird,plane,etc), 10 categories
y_test = to_categorical(y_test, 10)

for i in range(3):
    # show an example image from MNIST
    example = random.randint(0,49_999)
    plt.imshow(x_train[example][:,:,0]) #colon means not to change the value
    plt.title(class_labels[np.where(y_train[example]==1)[0][0]]) #set the title of the graph to be the object category
    #np.where finds all index values where the value in y_train[example] matches 1
    #that returns a tuple with just a numpy array, and in that numpy array are the integers for all of the index values (since I only want the first one, I used index of 0 for the array)
    #sending that index to class_labels outputs the object category
    plt.show()

batch_size = 64 #how many images to look at at a time; more in-depth data needs smaller batches
num_classes = 10 #bc theres 10 numbers
epochs = 10 #number of times through the data

# build the model
model_1 = tf.keras.models.Sequential( # 10 epochs: acc:75.0%, loss:70.0%, val_acc:75.2%, val_loss:71.2% - slightly overfit || 5 epochs: acc:69%, loss:87%, val_acc:72%, val_loss:79%
    [
        tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu', input_shape=input_shape),
        #convolutional 2D neural network building the edge detection: 32 kernels(filters), 5x5 p0x each; padding-> input+output same size - don't bother scanning the margins;
        #relu is a common shape; input_shape stays the same
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #same but finer comb through
        tf.keras.layers.MaxPool2D(), #flatten and reduce the size of things; converge data
        tf.keras.layers.Dropout(0.25), #turn off a random 25% of the nuerons(filters) to prevent overfitting (learning/infering instead of memorizing)
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #more filters than previous Conv2D
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Flatten(), #take all the info and flatten into one thing so that the dense can read it
        tf.keras.layers.Dense(num_classes, activation='softmax') #output layer; softmax activation gives probabilities that goes back to one-hot to get an estimate answer
    ]
)

model_2 = tf.keras.models.Sequential( # 5 epochs: acc:67%, loss:93%, val_acc:68%, val_loss:91% - lower acc, higher loss; will be re-adding layer 
    [
        tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), 
        tf.keras.layers.MaxPool2D(), 
        tf.keras.layers.Dropout(0.25), 
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), 
        # tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(num_classes, activation='softmax') 
    ]
)

model_3 = tf.keras.models.Sequential( # 5 epochs: acc:65%, loss:97%, val_acc:68%, val_loss:89% - worse than before, will attempt removing 3rd maxpool layer
    [
        tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), 
        tf.keras.layers.MaxPool2D(), 
        tf.keras.layers.Dropout(0.25), 
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), 
        tf.keras.layers.MaxPool2D(), #testing adding new layer
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(num_classes, activation='softmax') 
    ]
)

model_4 = tf.keras.models.Sequential( # 5 epochs: acc:70%, loss:85%, val_acc:72%, val_loss:78% - similar to model 1 but slightly better; also minimally overfit
    # || 10 epochs: acc:75.4%, loss:70.2%, val_acc:75.1%, val_loss:70.8%
    # conclusion: about the same as model 1, negligible difference; will be going back to model 1 as the base
    [
        tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), 
        tf.keras.layers.MaxPool2D(), 
        tf.keras.layers.Dropout(0.25), 
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), 
        tf.keras.layers.MaxPool2D(), #testing moving layer from after both to in-between
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        # tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(num_classes, activation='softmax') 
    ]
)

model_5 = tf.keras.models.Sequential( # 5 epochs: acc:71%, loss:81%, val_acc:74%, val_loss:74% || 10 epochs: acc:77.3%, loss:65.2%, val_acc:76.8%, val_loss:66.9%
    # slightly better than model 1, also slightly overfit after 10
    # conclusion: will use as final model.
    [
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), #testing adjusting 5x5 p0x to 3x3 p0x
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), 
        tf.keras.layers.MaxPool2D(), 
        tf.keras.layers.Dropout(0.25), 
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape), 
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(num_classes, activation='softmax') 
    ]
)

model_lol = tf.keras.models.Sequential( # 5 epochs: acc:38%, loss:178%, val_acc:35%, val_loss:185% || 10 epochs: acc:39%, loss:175%, val_acc:36%, val_loss:183%
    # this wasn't expected to be accurate in the slightest, I was just curious how it would do
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]
)

# categorical_crossentropy is for one-hot, to force the prediction into a category; the metric for its answer is decided by which one has the highest accuracy
# if not one-hot, use softmax for loss instead
# common optimizer: Adam  - it's just an algorithm; idk whats the difference between these two 
# model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])
model_5.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

# model is trained on data from x- and y- train; batch size and epochs are inputed from vars defined earlier; x- and y- test are assigned to validate the model's accuracy
# assigning the model to variable(history) makes it easier to reference for the graphs
history = model_5.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# plot out training and validation accuraccy and loss - MatPlotLib
fig, ax = plt.subplots(2,1) #two plots at once; one on top and the other on the bottom; idk what the 'fig' at the start does tho

# loss is percent confidence; more loss means less confidence in the answer
ax[0].plot(history.history['loss'], color='b', label="Training Loss") #'b' = blue
ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss") #'r' = red
legend = ax[0].legend(loc='best', shadow=True) #loc = chose the best spot for it to go
ax[0].set_title("Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True) 
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")

plt.tight_layout() #idk what exactly this does.
plt.show()
# if the graph shows the training data cross the validation data, then the model is overfit

# predict the test data
test_loss, test_acc = model_5.evaluate(x_test, y_test) 
print('Test accuracy:', test_acc)

#generate the confusion matrix

# Predict the values from the testing dataset
Y_pred = model_5.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert testing observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1)
# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes) 

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# model summary
print('Model Summary:', model_5.summary())