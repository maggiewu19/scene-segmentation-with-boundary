import cv2
import numpy as np
import time 
from matplotlib import pyplot as plt

IMG_DIR = '../images/'

# white: 255
# black: 0 

def plotImage(original, updated): 
    plt.subplot(121),plt.imshow(original, cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(updated, cmap = 'gray')
    plt.title('Updated Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def loadImage(filename):
    return cv2.imread(filename, 1)

def getEdges(img):
    return cv2.Canny(img, 200, 200)

def loopPixels(edges, image): 
    def updateColors(x, y, color): 
        results = set() 
        for i in range(x-1, x+1): 
            for j in range(y-1, y+1): 
                if 0 <= i < len(image) and 0 <= j < len(image[0]):
                    results.add((i, j, color))

        return results 

    global_results = set() 
    for index, pixel in np.ndenumerate(edges):
        x, y = index
        if pixel == 255: 
            r, g, b = tuple(image[x, y])
            color = (max(r-20,0), max(g-20,0), max(b-20,0))
            global_results.update(updateColors(x, y, color))

    return global_results

def updateImage(filename):
    img = loadImage(filename)
    edges = getEdges(img)

    results = loopPixels(edges, img)
    for (i, j, color) in results: 
        img[i,j] = color

    return img 

def createFilter():
    kernel = np.zeros((9,9), np.float32)
    kernel[4,4] = 2.0

    boxFilter = np.ones((9,9), np.float32) / 81.0 
    return kernel - boxFilter

def sharpenImage(filename):
    img = loadImage(filename)
    kernel = createFilter()
    return cv2.filter2D(img, -1, kernel)

def saveImage(filename, img):
    cv2.imwrite(IMG_DIR + filename, img)

if __name__ == '__main__': 
    shells = loadImage(IMG_DIR + 'shells.png')
    new_shells = updateImage(IMG_DIR + 'shells.png')
    kernel_shells = sharpenImage(IMG_DIR + 'shells.png')

    plane = loadImage(IMG_DIR + 'plane.png')
    new_plane = updateImage(IMG_DIR + 'plane.png')
    kernel_plane = sharpenImage(IMG_DIR + 'plane.png')

    wine = loadImage(IMG_DIR + 'wine.png')
    new_wine = updateImage(IMG_DIR + 'wine.png')
    kernel_wine = sharpenImage(IMG_DIR + 'wine.png')

    test = loadImage(IMG_DIR + 'test.png')
    new_test = updateImage(IMG_DIR + 'test.png')
    kernel_test = sharpenImage(IMG_DIR + 'test.png')

    room = loadImage(IMG_DIR + 'room.png')
    new_room = updateImage(IMG_DIR + 'room.png')
    kernel_room = sharpenImage(IMG_DIR + 'room.png')

    lot = loadImage(IMG_DIR + 'lot.png')
    new_lot = updateImage(IMG_DIR + 'lot.png')
    kernel_lot = sharpenImage(IMG_DIR + 'lot.png')

    saveImage('darken_shells.png', new_shells)
    saveImage('darken_plane.png', new_plane)
    saveImage('darken_test.png', new_test)
    saveImage('darken_wine.png', new_wine)
    saveImage('darken_room.png', new_room)
    saveImage('darken_lot.png', new_lot)

    saveImage('kernel_shells.png', kernel_shells)
    saveImage('kernel_plane.png', kernel_plane)
    saveImage('kernel_test.png', kernel_test)
    saveImage('kernel_wine.png', kernel_wine)
    saveImage('kernel_room.png', kernel_room)
    saveImage('kernel_lot.png', kernel_lot)


