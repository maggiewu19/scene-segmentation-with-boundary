import cv2
import numpy as np
import time 
from matplotlib import pyplot as plt


# white: 255
# black: 0 

img = cv2.imread('test.jpg',1)
image = cv2.imread('test.jpg',1)
edges = cv2.Canny(img,100,200)

def loopPixels(edges, image): 
  def updateColors(x, y, color): 
    results = set() 
    for i in range(x-1, x+2): 
      for j in range(y-1, y+2): 
        if 0 <= i < len(image) and 0 <= j < len(image[0]):
          results.add((i, j))

    return results 

  global_results = set() 
  for index, pixel in np.ndenumerate(edges):
    x, y = index
    if pixel == 255: 
      color = image[x, y] 
      global_results.update(updateColors(x, y, color))

  return global_results


results = loopPixels(edges, img)
for (i, j) in results: 
  image[i,j] = 0 

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(image,cmap = 'gray')
plt.title('Updated Image'), plt.xticks([]), plt.yticks([])

plt.show()