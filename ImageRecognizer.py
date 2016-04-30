from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import sys

#Averages out an image array so there is only black and white
def threshold(imageArray):

    balanceArray = []
    thresholdedArray = imageArray
    for row in imageArray:
        for pixel in row:
            average = np.mean(pixel[:3])
            balanceArray.append(average)

    balance = np.mean(balanceArray)
    for row in thresholdedArray:
        for pixel in row:
            if np.mean(pixel[:3]) > balance:
                pixel[0] = 255
                pixel[1] = 255
                pixel[2] = 255
            else:
                pixel[0] = 0
                pixel[1] = 0
                pixel[2] = 0
    return thresholdedArray

#Takes a threshold'd image array and returns a new 1d array with 0s where white was and 1s where black was
def featurize(imageArray):
    array = []
    for row in imageArray:
        for pixel in row:
            if pixel[0] == 255:
                array.append(0)
            else:
                array.append(1)
    return array

def main():
    #Featurizing image groups 1-5
    # data = [[], [], [], [], []]
    # for i in range(1,86):
    #     filename = "images/1/" + str(i) + ".jpg"
    #     i = Image.open(filename)
    #     i.thumbnail((50, 50), Image.ANTIALIAS)
    #     iar = np.array(i)
    #     iar = threshold(iar)
    #     data[0].append(featurize(iar))
    # for i in range(1,73):
    #     filename = "images/2/" + str(i) + ".jpg"
    #     i = Image.open(filename)
    #     i.thumbnail((50, 50), Image.ANTIALIAS)
    #     iar = np.array(i)
    #     iar = threshold(iar)
    #     data[1].append(featurize(iar))
    # for i in range(1,89):
    #     filename = "images/3/" + str(i) + ".jpg"
    #     i = Image.open(filename)
    #     i.thumbnail((50, 50), Image.ANTIALIAS)
    #     iar = np.array(i)
    #     iar = threshold(iar)
    #     data[2].append(featurize(iar))
    # for i in range(1,82):
    #     filename = "images/4/" + str(i) + ".jpg"
    #     i = Image.open(filename)
    #     i.thumbnail((50, 50), Image.ANTIALIAS)
    #     iar = np.array(i)
    #     iar = threshold(iar)
    #     data[3].append(featurize(iar))
    # for i in range(1,88):
    #     filename = "images/5/" + str(i) + ".jpg"
    #     i = Image.open(filename)
    #     i.thumbnail((50, 50), Image.ANTIALIAS)
    #     iar = np.array(i)
    #     iar = threshold(iar)
    #     data[4].append(featurize(iar))
    # np.save("featurized_data.npy", data)

    #Loading in already featurized data for time saving
    data = np.load("featurized_data.npy")

    #Building training data, leaving last 10 off as potential testing data
    X = data[0][:-10]
    y = [1] * 75
    X.extend(data[1][:-10])
    y.extend([2] * 62)
    X.extend(data[2][:-10])
    y.extend([3] * 78)
    X.extend(data[3][:-10])
    y.extend([4] * 71)
    X.extend(data[4][:-10])
    y.extend([5] * 77)

    #Fitting data with python support vector machine
    clf = svm.SVC(gamma=0.001, C=100)
    clf.fit(X, y)

    #Featutizing image from system argument and attempting to predict its shape
    imageLocation = sys.argv[1]
    i = Image.open(imageLocation)
    i.thumbnail((50, 50), Image.ANTIALIAS)
    iar = np.array(i)
    iar = threshold(iar)
    featurizedImage = featurize(iar)
    prediction = clf.predict(np.reshape(featurizedImage, (1,-1)))
    if prediction[0] == 1:
        print "Smile"
    elif prediction[0] == 2:
        print "Hat"
    elif prediction[0] == 3:
        print "Hash"
    elif prediction[0] == 4:
        print "Heart"
    else:
        print "Dollar"

    #For testing overall accuracy
    # outfile = open("outt.txt", "w")
    # correct = 0
    # wrong = 0
    # total = 0
    # for i in data:
    #     correct += 1
    #     for j in i:
    #         total += 1
    #         prediction = clf.predict(np.reshape(j, (1,-1)))
    #         if prediction != correct:
    #             wrong += 1
    #         outfile.write(str(prediction))
    # print wrong
    # print total

main()
