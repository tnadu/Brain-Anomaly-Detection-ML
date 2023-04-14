import numpy as np      # for matrix operations
import pandas as pd     # convenient csv operations
import cv2              # image reading, resizing and augmentation
import random           # for random angle generation
# for mixing the validation and train set into a common pool
from sklearn.model_selection import train_test_split
from sklearn import preprocessing       # normalization module
# the three models
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
# various required scoring measurements
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# values to tweak
randomSplit = False
augmented = False
resolution = (120, 120)
center = (resolution[0] // 2, resolution[0] // 2)
flattenedResolution = resolution[0] ** 2
numberOfBins = 4        # for naive bayes
K = 9                   # for KNN
p = 1                   # distance used in KNN
C = 0.5                 # for SVM
distanceType = 'l1'     # distance used in SVM
normalizer = preprocessing.Normalizer()


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
# making the array 2-dimensional, as required by the model
trainImages = np.reshape(trainImages, (-1, flattenedResolution))
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
    image = cv2.imread(f'../data/{imageIndex:06d}.png', 0)
    # resizing the image
    image = cv2.resize(image, resolution)
    testImages.append(image)

print('Test images loaded')
# converting to np array
testImages = np.array(testImages)
# making the array 2-dimensional, as required by the model
testImages = np.reshape(testImages, (-1, flattenedResolution))
print('Test images converted and normalized\n')


# TRAINING AND PREDICTING
# KNN
print(f'\nInitializing KNN training for K={K}')
knnClasifier = KNeighborsClassifier(n_neighbors=K, p=p)   # instantiating model
knnClasifier.fit(trainImages, trainLabels)                # fitting the training data
print('Training complete')

print('Initializing scoring on validation set')
knnValidationPredictions = knnClasifier.predict(validationImages)
print('Predicting test set initiated')
knnPredictions = knnClasifier.predict(testImages)


# Naive Bayes
# defining a set of bins to put the 0-255 values into
bins = np.linspace(start=0, stop=255, num=numberOfBins)

# converting the data
nbTrainImages = np.digitize(trainImages, bins) - 1
nbValidationImages = np.digitize(validationImages, bins) - 1
nbTestImages = np.digitize(testImages, bins) - 1

print(f'Initializing Naive Bayes training for num-bins={numberOfBins}')
nbClassifier = MultinomialNB()                  # instantiating model
nbClassifier.fit(trainImages, trainLabels)      # fitting the training data
print('Training complete')

print('Initializing scoring on validation set')
nbValidationPredictions = nbClassifier.predict(validationImages)
print('Predicting test set initiated')
nbPredictions = nbClassifier.predict(testImages)

# free up unnecessary memory
del nbTrainImages, nbValidationImages, nbTestImages


# SVM
# normalizing the images using the specified distance type
trainImages = normalizer.fit_transform(trainImages, distanceType)
# normalizing the images using the specified distance type
validationImages = normalizer.fit_transform(validationImages, distanceType)
# normalizing the images using the specified distance type
testImages = normalizer.fit_transform(testImages, distanceType)

print(f'\nInitializing SVM training for C={C}')
# instantiating model, making sure to initialize
# weights according to the frequency of each class
svmClassifier = LinearSVC(C=C, class_weight='balanced', dual=False)
# fitting the training data
svmClassifier.fit(trainImages, trainLabels)
print('Training complete')

print('Initializing scoring on validation set')
svmValidationPredictions = svmClassifier.predict(validationImages)
print('Predicting test set initiated')
svmPredictions = svmClassifier.predict(testImages)


# MAJORITY VOTING
validationPredictions = [1 if nbValidationPredictions[i] + knnValidationPredictions[i] + svmValidationPredictions[i] > 1 else 0 for i in range(len(validationLabels))]
predictions = [1 if nbPredictions[i] + knnPredictions[i] + svmPredictions[i] > 1 else 0 for i in range(len(testFileNames))]

# generating a string with relevant metric results
reportedMetrics = f'Random split: {randomSplit}\n' \
                  f'Augmentation: {augmented}\n' \
                  f'Resolution: {resolution}\n' \
                  f'Number of bins: {numberOfBins}\n' \
                  f'K: {K}\n' \
                  f'Distance type KNN: {p}\n' \
                  f'C: {C}\n' \
                  f'Distance type SVM: {distanceType}\n' \
                  f'Normalization: {distanceType}\n' \
                  f'Accuracy: {accuracy_score(validationLabels, validationPredictions)}\n' \
                  f'Precision: {precision_score(validationLabels, validationPredictions)}\n' \
                  f'Recall: {recall_score(validationLabels, validationPredictions)}\n' \
                  f'Confusion matrix:\n{confusion_matrix(validationLabels, validationPredictions)}\n\n'

print(reportedMetrics)
with open('nb-knn-svm-majority-reports.txt', 'a') as reportFile:
    reportFile.write(reportedMetrics)

# converting to pd format in order to conveniently
# save the test set predictions to a csv file
predictions = pd.DataFrame({"id": testFileNames, "class": predictions})
predictions.to_csv(f'nb-bins-{numberOfBins}-knn-k-{K}-c-{C}.csv', index=None)
