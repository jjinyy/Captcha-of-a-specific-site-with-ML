import cv2
import numpy as np

img = cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE)
img.shape
#cv2.inshow()


'''
#i=1
#for i in 50:
img = cv2.imread('images/1.jpg', cv2.IMREAD_GRAYSCALE)
img.shape

plt.imshow(img)
#plt.xticks([]) # x축 눈금
#plt.yticks([]) # y축 눈금
plt.show()

img = cv2.imread('images/2.jpg', cv2.IMREAD_GRAYSCALE)
img.shape

plt.imshow(img)
#plt.xticks([]) # x축 눈금
#plt.yticks([]) # y축 눈금
plt.show()

img = cv2.imread('images/3.jpg', cv2.IMREAD_GRAYSCALE)
img.shape

plt.imshow(img)
#plt.xticks([]) # x축 눈금
#plt.yticks([]) # y축 눈금
plt.show()

img = cv2.imread('images/4.jpg', cv2.IMREAD_GRAYSCALE)
img.shape

plt.imshow(img)
#plt.xticks([]) # x축 눈금
#plt.yticks([]) # y축 눈금
plt.show()

#-*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import sys

FNAME = 'digits.npz'

def machineLearning():
    img = cv2.imread('images/digits.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
    x = np.array(cells)
    train = x[:,:].reshape(-1,400).astype(np.float32)

    k = np.arange(10)
    train_labels = np.repeat(k,500)[:,np.newaxis]

    np.savez(FNAME,train=train,train_labels = train_labels)

def resize20(pimg):
    img = cv2.imread(pimg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayResize = cv2.resize(gray,(20,20))
    ret, thresh = cv2.threshold(grayResize, 125, 255,cv2.THRESH_BINARY_INV)

    cv2.imshow('num',thresh)
    return thresh.reshape(-1,400).astype(np.float32)

def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

    return train, train_labels

def checkDigit(test, train, train_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    return result

if __name__ == "__main__":

    #machineLearning()

    train, train_labels = loadTrainData(FNAME)

    saveNpz = False
    for fname in glob.glob('images/num*.png'):
        test = resize20(fname)
        result = checkDigit(test, train, train_labels)

        print ("result")

        k = cv2.waitKey(0)

        if k > 47 and k<58:
            saveNpz = True
            train = np.append(train, test, axis=0)
            newLabel = np.array(int(chr(k))).reshape(-1,1)
            train_labels = np.append(train_labels, newLabel,axis=0)


    cv2.destroyAllWindows()
    if saveNpz:
        np.savez(FNAME,train=train, train_labels=train_labels)
   # else:
   #    print ("unknow option")
   '''





















