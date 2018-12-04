import cv2
import numpy as np
import time 
from matplotlib import pyplot as plt

IMG_DIR = '../images/'

ORIGINAL_DIR = 'original/'
DARKEN_DIR = 'darken/'
KERNEL_DIR = 'kernel/'
NEW_DIR = 'new/'
PURE_DARKEN_DIR = 'pure_darken/'
DARKEN_KERNEL_DIR = 'darken_kernel/'
KERNEL_DARKEN_DIR = 'kernel_darken/'

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

def loopPixels(edges, image, size, darken): 
    def updateColors(x, y, color): 
        results = set() 
        x_range = range(x+size[0], x+size[1])
        y_range = range(y+size[0], y+size[1])
        for i in x_range: 
            for j in y_range: 
                if 0 <= i < len(image) and 0 <= j < len(image[0]):
                    results.add((i, j, color))

        return results 

    global_results = set() 
    for index, pixel in np.ndenumerate(edges):
        x, y = index
        if pixel == 255: 
            r, g, b = tuple(image[x, y])
            color = (max(r-50,0), max(g-50,0), max(b-50,0)) if darken else (r,g,b)
            global_results.update(updateColors(x, y, color))

    return global_results

def updateImage(filename, size, darken=False):
    img = loadImage(filename)
    edges = getEdges(img)

    results = loopPixels(edges, img, size, darken)
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

def saveAllImages(filename):
    img = loadImage(IMG_DIR + ORIGINAL_DIR + filename)
    new_img = updateImage(IMG_DIR + ORIGINAL_DIR + filename, (-1, 1))
    darken_img = updateImage(IMG_DIR + ORIGINAL_DIR + filename, (-1, 1), darken=True)
    kernel_img = sharpenImage(IMG_DIR + ORIGINAL_DIR + filename)
    pure_darken_img = updateImage(IMG_DIR + ORIGINAL_DIR + filename, (0, 1), darken=True)

    saveImage(IMG_DIR + NEW_DIR + 'new_' + filename, new_img)
    saveImage(IMG_DIR + DARKEN_DIR + 'darken_' + filename, darken_img)
    saveImage(IMG_DIR + KERNEL_DIR + 'kernel_' + filename, kernel_img)
    saveImage(IMG_DIR + PURE_DARKEN_DIR + 'pure_darken_' + filename, pure_darken_img)

    kernel_darken_img = updateImage(IMG_DIR + KERNEL_DIR + 'kernel_' + filename, (0, 1), darken=True)
    darken_kernel_img = sharpenImage(IMG_DIR + PURE_DARKEN_DIR + 'pure_darken_' + filename)

    saveImage(IMG_DIR + KERNEL_DARKEN_DIR + 'kernel_darken_' + filename, kernel_darken_img)
    saveImage(IMG_DIR + DARKEN_KERNEL_DIR + 'darken_kernel_' + filename, darken_kernel_img)

if __name__ == '__main__': 
    images = ['shells.png', 'plane.png', 'wine.png', 'room.png', 'lot.png']

    for img in images: 
        saveAllImages(img)


