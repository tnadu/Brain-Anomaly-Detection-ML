import numpy as np      # for matrix operations
import pandas as pd     # convenient csv operations
import cv2              # image reading, resizing and augmentation
import random           # for random angle generation
from sklearn.naive_bayes import MultinomialNB       # the model in use
# various required scoring measurements
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# values to tweak
augmented = True
resolution = (120, 120)
center = (resolution[0] // 2, resolution[0] // 2)
resolution = (120, 120)
flattenedResolution = resolution[0] ** 2


# reading the data and separating it into
# file names and the associated labels
data = pd.read_csv('../train_labels.txt')
fileNames = data['id'].tolist()[:17001]
labels = data['class'].tolist()[:17001]

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
    currentImage = cv2.imread(f'../data/{imageIndex:06d}.png', 0)
    # resizing the image
    currentImage = cv2.resize(currentImage, resolution)
    trainImages.append(currentImage)

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
            currentImage = cv2.imread(f'../data/{imageIndex:06d}.png', 0)
            # resizing the image
            currentImage = cv2.resize(currentImage, resolution)
            # applying the rotation
            currentImage = cv2.warpAffine(currentImage, rotationMatrix, resolution)
            trainImages.append(currentImage)

print('Train images loaded')
# converting to np array
trainImages = np.array(trainImages)
# making the array 2-dimensional, as required by the model
trainImages = np.reshape(trainImages, (-1, flattenedResolution))
print('Train images converted and normalized\n')


# Validation set
validationImages = []

# enumerating the file name list to get the
# current index for monitoring purposes
for i, imageIndex in enumerate(fileNameValidate):
    if i in range(0, 2000, 500):
        print(f'Loading validation image #{i}')

    # loading the image as grayscale
    currentImage = cv2.imread(f'../data/{imageIndex:06d}.png', 0)
    # resizing the image
    currentImage = cv2.resize(currentImage, resolution)
    validationImages.append(currentImage)

print('Validation images loaded')
# converting to np array
validationImages = np.array(validationImages)
# making the array 2-dimensional, as required by the model
validationImages = np.reshape(validationImages, (-1, flattenedResolution))
print('Validation images converted and normalized\n')


# Test set
testImages = []

# enumerating the file name list to get the
# current index for monitoring purposes
for i, imageIndex in enumerate(range(17001, 22150)):
    if i in range(0, 5000, 1000):
        print(f'Loading test image #{i}')

    # loading the image as grayscale
    currentImage = cv2.imread(f'../data/{imageIndex:06d}.png', 0)
    # resizing the image
    currentImage = cv2.resize(currentImage, resolution)
    testImages.append(currentImage)

print('Test images loaded')
# converting to np array
testImages = np.array(testImages)
# making the array 2-dimensional, as required by the model
testImages = np.reshape(testImages, (-1, flattenedResolution))
print('Test images converted and normalized\n')


for numberOfBins in range(2, 12, 2):
    # DATA PREPROCESSING
    # defining a set of bins to put the 0-255 values into
    bins = np.linspace(start=0, stop=255, num=numberOfBins)

    # converting the data
    currentTrainImages = np.digitize(trainImages, bins) - 1
    currentValidationImages = np.digitize(validationImages, bins) - 1
    testImages = np.digitize(testImages, bins) - 1


    # TRAINING AND PREDICTING
    print(f'Initializing Naive Bayes training for num-bins={numberOfBins}')
    nbClassifier = MultinomialNB()                         # instantiating model
    nbClassifier.fit(currentTrainImages, trainLabels)      # fitting the training data
    print('Training complete')

    print('Initializing scoring on validation set')
    validationPredictions = nbClassifier.predict(currentValidationImages)
    # generating a string with relevant metric results
    reportedMetrics = f'Augmentation: {augmented}\n' \
                      f'Resolution: {resolution}\n' \
                      f'Number of bins: {numberOfBins}\n' \
                      f'Accuracy: {accuracy_score(validationLabels, validationPredictions)}\n' \
                      f'Precision: {precision_score(validationLabels, validationPredictions)}\n' \
                      f'Recall: {recall_score(validationLabels, validationPredictions)}\n' \
                      f'Confusion matrix:\n{confusion_matrix(validationLabels, validationPredictions)}\n\n'

    print(reportedMetrics)
    with open('nb-reports.txt', 'a') as reportFile:
        reportFile.write(reportedMetrics)

    print('Predicting test set initiated')
    predictions = nbClassifier.predict(testImages)
    # converting to pd format in order to conveniently
    # save the test set predictions to a csv file
    predictions = pd.DataFrame({"id": testFileNames, "class": predictions})
    predictions.to_csv(f'nb-{numberOfBins}.csv', index=None)
