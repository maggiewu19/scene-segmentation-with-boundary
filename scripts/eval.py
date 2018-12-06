import cv2
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage import data, img_as_float 

IMG_DIR = '../images/'
CROPPED_DIR = 'cropped/'
GROUND_TRUTH_DIR = 'ground_truth/ground_truth_'

def pixel_wise_mse(img, ground_truth):
    img = img_as_float(img)
    ground_truth = img_as_float(ground_truth)

    num_pixels = len(img) * len(img[0])

    img_se = mse(img, ground_truth)

    return float(img_se) / num_pixels

def pixel_wise_accuracy(img, ground_truth):
    img = img_as_float(img)
    ground_truth = img_as_float(ground_truth)

    num_pixels = len(img) * len(img[0])

    img_loss = loss(img, ground_truth)
    img_loss[img_loss >= 0.5] = 1
    total_loss = np.sum(img_loss)

    return float(total_loss) / num_pixels

def ssim_measure(img, ground_truth):
    img = img_as_float(img)
    ground_truth = img_as_float(ground_truth)

    img_ssim = ssim(ground_truth, img, data_range=img.max()-img.min(), multichannel=True)
    return img_ssim 

def loss(x, y): 
    return abs(x-y)

def mse(x, y): 
    return np.linalg.norm(x-y)

def process_metrics(images, types):
    MSE = dict()
    SSIM = dict() 
    ZERO_ONE = dict() 

    for t in types:
        if t == '': 
            MSE['baseline'] = list() 
            SSIM['baseline'] = list() 
            ZERO_ONE['baseline'] = list() 
        else: 
            MSE[t[0:-1]] = list() 
            SSIM[t[0:-1]] = list() 
            ZERO_ONE[t[0:-1]] = list() 

    GROUND_TRUTH_FOLDER = IMG_DIR + GROUND_TRUTH_DIR
    IMG_FOLDER = IMG_DIR + CROPPED_DIR

    # cropped
    for i in images:
        for t in types: 
            img = t + i

            # get paths for image selected
            img_path = IMG_FOLDER + img
            ground_truth_path = GROUND_TRUTH_FOLDER + i 

            img = Image.open(img_path)
            ground_truth = Image.open(ground_truth_path)

            pixel_mse = pixel_wise_mse(img, ground_truth)
            pixel_acc = pixel_wise_accuracy(img, ground_truth)
            ssim_val = ssim_measure(img, ground_truth)

            if t == '':
                MSE['baseline'].append(pixel_mse)
                SSIM['baseline'].append(ssim_val)
                ZERO_ONE['baseline'].append(pixel_acc)
            else:
                MSE[t[0:-1]].append(pixel_mse)
                SSIM[t[0:-1]].append(ssim_val)
                ZERO_ONE[t[0:-1]].append(pixel_acc)

        print ('finished ' + i[0:-4])

    return MSE, SSIM, ZERO_ONE

def getMeanStd(d, types):
    avg, sd = list(), list()
    for t in types: 
        t = 'baseline' if t == '' else t[0:-1]

        print (t, np.mean(d[t]), np.std(d[t]))
        avg.append(np.mean(d[t]))
        sd.append(np.std(d[t]))

    return avg, sd 

def plotErrorStd(avg, sd, ylabel, title): 
    tick_name = ['B', 'T', 'DT', 'PD', 'K', 
                'C-50', 'C-100', 'DK', 'KD']

    x_pos = np.arange(len(tick_name))
    CTEs = avg
    error = sd

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_name)
    ax.set_title(title)
    ax.yaxis.grid(True)

    # Show the figure
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    IMAGES = ['bread.png', 'house.png', 'kitchen_table.png', 'mountain.png',
        'park.png', 'pool.png', 'shoes.png', 'sofa.png', 'street.png']

    TYPES = ['', 'new_', 'darken_', 'pure_darken_', 'kernel_', 
    'contrast_50_', 'contrast_100_', 'darken_kernel_', 'kernel_darken_']

    MSE, SSIM, ZERO_ONE = process_metrics(IMAGES, TYPES)

    print ('--- MSE ---')
    mse_avg, mse_sd = getMeanStd(MSE, TYPES)

    print ('--- SSIM ---')
    ssim_avg, ssim_sd = getMeanStd(SSIM, TYPES)

    print ('--- 0/1 LOSS ---')
    zero_one_avg, zero_one_sd = getMeanStd(ZERO_ONE, TYPES)

    ylabel = 'MSE'
    title = 'Mean Squared Error vs. Techniques'
    plotErrorStd(mse_avg, mse_sd, ylabel, title)

    ylabel = 'SSIM'
    title = 'Structural Similarity Index Measure vs. Techniques'
    plotErrorStd(ssim_avg, ssim_sd, ylabel, title)

    ylabel = '0/1 Loss'
    title = '0/1 Loss vs. Techniques'
    plotErrorStd(zero_one_avg, zero_one_sd, ylabel, title)


