import numpy as np

import os
from dataclasses import dataclass

import skimage
import skimage.io as skiIO
import skimage.color as skiColor

import cv2

# Data class to hold inImg's and tarImg's
@dataclass()
class ImgSet:
    inImg: np.ndarray
    tarImg: np.ndarray
    resImg: np.ndarray
    num: str
    def __init__(self,inImg:np.ndarray,tarImg:np.ndarray,num:str):
        self.inImg = inImg
        self.tarImg = tarImg
        self.num = num

    def setResImg(self,resImg:np.ndarray):
        self.resImg = resImg

def colorTransform(inImg:np.ndarray, tarImg:np.ndarray):

    # Transform rgb images to lab images
    inImg = skiColor.rgb2lab(inImg)
    tarImg = skiColor.rgb2lab(tarImg)
    # Calculate means and standard deviations for lab channels

    inChs = [inImg[:, :, 0], inImg[:, :, 1], inImg[:, :, 2]]  # inImg channels
    tarChs = [tarImg[:, :, 0], tarImg[:, :, 1], tarImg[:, :, 2]]  # tarImg channels

    resChs = inChs.copy()

    inMeans = [np.mean(inChs[0]), np.mean(inChs[1]), np.mean(inChs[2])]  # inImg channel means
    tarMeans = [np.mean(tarChs[0]), np.mean(tarChs[1]), np.mean(tarChs[2])]  # tarImg channel means

    inSds = [np.std(inChs[0]), np.std(inChs[1]), np.std(inChs[2])]  # inImg channels standard deviation
    tarSds = [np.std(tarChs[0]), np.std(tarChs[1]), np.std(tarChs[2])]  # tarImg channels standard deviation

    # Subtract means from data points
    for i in range(0, 3):
        resChs[i] = inChs[i] - inMeans[i]

    # Scale new data points
    for i in range(0, 3):
        resChs[i] = resChs[i] * (tarSds[i]/inSds[i])

    # Add averages of target
    for i in range(0,3):
        resChs[i] = resChs[i] + tarMeans[i]


    for i in range(0,  len(resChs[0])):
        for j in range(0,  len(resChs[0][0])):

            if(resChs[0][i][j] < 0) : resChs[0][i][j] = 0
            if(resChs[0][i][j] > 100) : resChs[0][i][j] = 100

            if (resChs[1][i][j] < -128): resChs[1][i][j] = -128
            if (resChs[1][i][j] > 127): resChs[1][i][j] = 127

            if (resChs[2][i][j] < -128): resChs[2][i][j] = -128
            if (resChs[2][i][j] > 127): resChs[2][i][j] = 127

    # print("AFTER")
    # print("MAX")
    # print(np.max(resChs[0]))
    # print(np.max(resChs[1]))
    # print(np.max(resChs[2]))
    # print("MIN")
    # print(np.min(resChs[0]))
    # print(np.min(resChs[1]))
    # print(np.min(resChs[2]))
    # print("==================================")


    resImg = np.dstack([resChs[0],resChs[1],resChs[2]])

    resImg = skiColor.lab2rgb(resImg)

    resImg = resImg * 255

    resImg = np.ndarray.astype(resImg, dtype="uint8")

    # print("AFTER AFTER")
    # print("MAX")
    # print(np.max(resImg[:,:,0]))
    # print(np.max(resImg[:,:,1]))
    # print(np.max(resImg[:,:,2]))
    # print("MIN")
    # print(np.min(resImg[:,:,0]))
    # print(np.min(resImg[:,:,1]))
    # print(np.min(resImg[:,:,2]))
    # print("==================================")


    return resImg


def showImg(img:np.ndarray):
    skiIO.imshow(img)
    skiIO.show()


def main():

    fileNames = os.listdir("./data")

    imgSetList = []

    # Load images from folder
    for fileName in fileNames:
        if(fileName.split("_")[0] == "in"):
            numStr = fileName.split(".")[0].split("_")[1]
            inImg = skiIO.imread(("./data/" + "in_" + numStr + ".png"))
            tarImg = skiIO.imread(("./data/" + "tar_" + numStr + ".png"))
            imgSetList.append(ImgSet(inImg, tarImg, numStr))

    for imgSet in imgSetList:

        resImg = colorTransform(imgSet.inImg, imgSet.tarImg)

        skiIO.imsave("./results/res_" + imgSet.num + ".png", resImg)


if __name__ == '__main__':
    main()
