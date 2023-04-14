import numpy as np      # for matrix operations
import pandas as pd     # convenient csv operations
import cv2              # image reading, resizing and augmentation
import random           # for random angle generation
from tensorflow import keras    # used in building the CNN
# for mixing the validation and train set into a common pool
from sklearn.model_selection import train_test_split
# various required scoring measurements
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# used for plotting the accuracy and loss through epochs
from matplotlib import pyplot as plt


# values to tweak
randomSplit = True
augmented = True
resolution = (120, 120)
center = (resolution[0] // 2, resolution[0] // 2)
learningRate = 0.00001
filterSize = 3     # convolutional filter size


# reading the data and separating it into
# file names and the associated labels
data = pd.read_csv('../train_labels.txt')
fileNames = data['id'].tolist()[:17001]
labels = data['class'].tolist()[:17001]

if randomSplit:
    # from a common pool consisting of the train and validation sets,
    # a fair split between the normal and abnormal samples is done
    fileNameTrain, fileNameValidate, trainLabels, validationLabels = train_test_split(fileNames, labels, train_size=0.8)
else:
    # splitting the aforementioned data set
    # into a training and a validation set
    fileNameTrain = fileNames[:15001]
    fileNameValidate = fileNames[15000:]
    trainLabels = labels[:15001]
    validationLabels = labels[15000:]

# building a list of strings corresponding to the test set
testFileNames = [f'{i:06d}' for i in range(17001, 22150)]


# DATA LOADING
# Train set
trainImages = []

# enumerating the file name list to get the
# current index for monitoring purposes
for i, imageIndex in enumerate(fileNameTrain):
    if i in range(0, 12000, 1000):
        print(f'Loading train image #{i}')

    # loading the image as grayscale
    image = cv2.imread(f'../data/{imageIndex:06d}.png', 0)
    # resizing the image
    image = cv2.resize(image, resolution)
    trainImages.append(image)

# augmenting the abnormal images by randomly
# rotating them between -10 and 10 degrees
if augmented:
    # filtering the abnormal images
    fileNameTrainAbnormal = [fileNameTrain[i] for i, value in enumerate(trainLabels) if value == 1]
    # extending the label list to accommodate the augmented images
    trainLabels.extend([1] * 7 * len(fileNameTrainAbnormal))

    # augmenting all abnormal images 7 times yields
    # a roughly equal normal-abnormal sample ratio
    for count in range(1, 8):
        print(f'Loading augmented train images, iteration #{count}')

        for i, imageIndex in enumerate(fileNameTrainAbnormal):
            # generating a random rotation matrix
            rotationMatrix = cv2.getRotationMatrix2D(center, random.randint(-10, 10), 1)

            # loading the image as grayscale
            image = cv2.imread(f'../data/{imageIndex:06d}.png', 0)
            # resizing the image
            image = cv2.resize(image, resolution)
            # applying the rotation
            image = cv2.warpAffine(image, rotationMatrix, resolution)
            trainImages.append(image)

print('Train images loaded')
# converting to np array
trainImages = np.array(trainImages)
print('Train images converted and normalized\n')


# Validation set
validationImages = []

# enumerating the file name list to get the
# current index for monitoring purposes
for i, imageIndex in enumerate(fileNameValidate):
    if i in range(0, 3000, 500):
        print(f'Loading validation image #{i}')

    # loading the image as grayscale
    image = cv2.imread(f'../data/{imageIndex:06d}.png', 0)
    # resizing the image
    image = cv2.resize(image, resolution)
    validationImages.append(image)

print('Validation images loaded')
# converting to np array
validationImages = np.array(validationImages)
print('Validation images converted and normalized\n')


# Test set
testImages = []

# enumerating the file name list to get the
# current index for monitoring purposes
for i, imageIndex in enumerate(range(17001, 22150)):
    if i in range(0, 5000, 1000):
        print(f'Loading test image #{i}')

    # loading the image as grayscale
    image = cv2.imread(f'../data/{imageIndex:06d}.png', 0)
    # resizing the image
    image = cv2.resize(image, resolution)
    testImages.append(image)

print('Test images loaded')
# converting to np array
testImages = np.array(testImages)
print('Test images converted and normalized\n')


# TRAINING AND PREDICTING
# CNN - first attempt (simple)

# initializing the network as a sequence of layers
cnnClassifier = keras.Sequential()

# conv-maxPool input layer
cnnClassifier.add(keras.layers.Conv2D(8, filterSize, padding='same', input_shape=(224, 224, 1)))
cnnClassifier.add(keras.layers.MaxPooling2D())

# flattening the output in preparation for the fc layer
cnnClassifier.add(keras.layers.Flatten())

# output layer
cnnClassifier.add(keras.layers.Dense(2, activation='softmax'))

# standard optimizer
optimizerAdam = keras.optimizers.Adam(learning_rate=learningRate)
cnnClassifier.compile(optimizer=optimizerAdam, loss='categorical_crossentropy', metrics=['accuracy'])

historyToPlot = cnnClassifier.fit(
    trainImages,
    # converting labels to [x1, x2] arrays,
    # where x1 = 1 and x2 = 0 if label = 0
    # or x1 = 0 and x2 = 1 if label = 1
    keras.utils.to_categorical(trainLabels),
    epochs=15,
    batch_size=64,
    validation_data=(validationImages, keras.utils.to_categorical(validationLabels)),
    verbose=2,
)

# plotting
plt.plot(historyToPlot.history['accuracy'])
plt.plot(historyToPlot.history['loss'])
plt.plot(historyToPlot.history['val_accuracy'])
plt.plot(historyToPlot.history['val_loss'])

plt.xlabel("epoch")
plt.ylabel("value")
plt.legend(['training accuracy', 'training loss', 'validation accuracy', 'validation loss'])
plt.title("Simple CNN architecture")

plt.savefig('cnn-simple.png', bbox_inches='tight')
plt.show()

print('Initializing scoring on validation set')
validationPredictions = np.argmax(cnnClassifier.predict(validationImages), axis=1)
# generating a string with relevant metric results
reportedMetrics = f'Random split: {randomSplit}\n' \
                      f'Augmentation: {augmented}\n' \
                      f'Resolution: {resolution}\n' \
                      f'Learning rate: {learningRate}\n' \
                      f'Accuracy: {accuracy_score(validationLabels, validationPredictions)}\n' \
                      f'Precision: {precision_score(validationLabels, validationPredictions)}\n' \
                      f'Recall: {recall_score(validationLabels, validationPredictions)}\n' \
                      f'Confusion matrix:\n{confusion_matrix(validationLabels, validationPredictions)}\n\n'

print(reportedMetrics)
with open('cnn-simple-reports.txt', 'a') as reportFile:
    reportFile.write(reportedMetrics)

print('Predicting test set initiated')
# the predictions consist of lists of 2 elements,
# corresponding to the score of each class;
# the highest scoring class is chosen
predictions = np.argmax(cnnClassifier.predict(testImages), axis=1)
# saving to a csv file
predictions = pd.DataFrame({"id": testFileNames, "class": predictions})
predictions.to_csv(f'cnn-simple.csv', index=None)


# CNN - second attempt (more complex)

# initializing the network as a sequence of layers
cnnClassifier = keras.Sequential()

# conv-conv-maxPool input layer
cnnClassifier.add(keras.layers.Conv2D(32, filterSize, padding='same', input_shape=(224, 224, 1)))
cnnClassifier.add(keras.layers.Conv2D(32, filterSize, activation='relu'))
cnnClassifier.add(keras.layers.MaxPooling2D())

# conv-conv-maxPool layer
cnnClassifier.add(keras.layers.Conv2D(64, filterSize, padding='same', activation='relu'))
cnnClassifier.add(keras.layers.Conv2D(64, filterSize, padding='same', activation='relu'))
cnnClassifier.add(keras.layers.MaxPooling2D())

# conv-conv-maxPool layer
cnnClassifier.add(keras.layers.Conv2D(128, filterSize, padding='same', activation='relu'))
cnnClassifier.add(keras.layers.Conv2D(128, filterSize, padding='same', activation='relu'))
cnnClassifier.add(keras.layers.MaxPooling2D())

# flattening the output in preparation for the fc layer
cnnClassifier.add(keras.layers.Flatten())

# extra fc layer
cnnClassifier.add(keras.layers.Dense(256, activation='relu'))
# output layer
cnnClassifier.add(keras.layers.Dense(2, activation='softmax'))

# standard optimizer
optimizerAdam = keras.optimizers.Adam(learning_rate=learningRate)
cnnClassifier.compile(optimizer=optimizerAdam, loss='categorical_crossentropy', metrics=['accuracy'])

historyToPlot = cnnClassifier.fit(
    trainImages,
    # converting labels to [x1, x2] arrays,
    # where x1 = 1 and x2 = 0 if label = 0
    # or x1 = 0 and x2 = 1 if label = 1
    keras.utils.to_categorical(trainLabels),
    epochs=15,
    batch_size=64,
    validation_data=(validationImages, keras.utils.to_categorical(validationLabels)),
    verbose=2,
)

# plotting
plt.plot(historyToPlot.history['accuracy'])
plt.plot(historyToPlot.history['loss'])
plt.plot(historyToPlot.history['val_accuracy'])
plt.plot(historyToPlot.history['val_loss'])

plt.xlabel("epoch")
plt.ylabel("value")
plt.legend(['training accuracy', 'training loss', 'validation accuracy', 'validation loss'])
plt.title("Complex CNN architecture")

plt.savefig('cnn-complex.png', bbox_inches='tight')
plt.show()

print('Initializing scoring on validation set')
validationPredictions = np.argmax(cnnClassifier.predict(validationImages), axis=1)
# generating a string with relevant metric results
reportedMetrics = f'Random split: {randomSplit}\n' \
                      f'Augmentation: {augmented}\n' \
                      f'Resolution: {resolution}\n' \
                      f'Learning rate: {learningRate}\n' \
                      f'Accuracy: {accuracy_score(validationLabels, validationPredictions)}\n' \
                      f'Precision: {precision_score(validationLabels, validationPredictions)}\n' \
                      f'Recall: {recall_score(validationLabels, validationPredictions)}\n' \
                      f'Confusion matrix:\n{confusion_matrix(validationLabels, validationPredictions)}\n\n'

print(reportedMetrics)
with open('cnn-complex-reports.txt', 'a') as reportFile:
    reportFile.write(reportedMetrics)

print('Predicting test set initiated')
# the predictions consist of lists of 2 elements,
# corresponding to the score of each class;
# the highest scoring class is chosen
predictions = np.argmax(cnnClassifier.predict(testImages), axis=1)
# saving to a csv file
predictions = pd.DataFrame({"id": testFileNames, "class": predictions})
predictions.to_csv(f'cnn-complex.csv', index=None)
