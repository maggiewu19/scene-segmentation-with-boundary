import cv2
import numpy as np
from edge_detection import *
import skimage.measure import structural_similarity as ssim

PREDICT_DIR = 'prediction_1/'

def crop_first_half(image):
    height, width = image.shape[:2]
    start_row, start_col = int(0), int(width*0.5)
    end_row, end_col = height, width
    cropped_img = image[start_row:end_row, start_col:end_col]
    return cropped_img

def same_size(predicted, actual):
    height, width = acutal.shape[:2]
    return predicted.resize((height, width))

def pixel_wise_accuracy(predicted, actual):
    pass

def mse(predicted, actual):
    err = np.sum((predicted.astype("float") - actual.astype("float"))**2)
    err /= float(predicted.shape[0] * predicted.shape[1])
    return err
    
    

if __name__ == '__main__':
    images = ['bedroom.jpg', 'classroom.jpg', 'dentist_office.jpg', 'elevator.jpg',
              'lot.png', 'room.png', 'ice_skating.jpg', 'kitchen.jpg', 'lake.jpg',
              'movie_theater.jpg', 'sail.jpg', 'office.jpg']
    
    # predicted
    for filename in images:
        img = loadImage(IMG_DIR + PREDICTION_DIR + filename)

    # get ground truth for image selected


    # eval: pixel-by-pixel difference with a threshold

