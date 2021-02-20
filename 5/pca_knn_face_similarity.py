import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import cv2
import os


def main():
    me = cv2.imread("cankut.jpg", cv2.IMREAD_GRAYSCALE)
    me = me.flatten()

    images = []
    labels = []
    names = []

    for folder in os.listdir("VGGFace-subset"):
        names.append(folder)
        for name in os.listdir("VGGFace-subset\\" + folder):
            img = cv2.imread("VGGFace-subset\\" + folder + "\\" + name, cv2.IMREAD_GRAYSCALE)
            images.append(img.flatten())
            labels.append(folder)

    pca = PCA(n_components=100)
    pca.fit(images)
    images = pca.transform(images)
    me = pca.transform([me])

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(images, labels)

    predictions = knn.predict_proba(me)

    print("Most similar celebrities")
    for i in range(len(names)):
        if predictions[0, i] == predictions.max():
            print("Similarity with " + names[i] + " : " + str(predictions[0, i]))


if __name__ == "__main__":
    main()