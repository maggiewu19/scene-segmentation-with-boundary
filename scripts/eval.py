import cv2
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage import data, img_as_float 

IMG_DIR = '../images/'
CROPPED_DIR = 'cropped/'
GROUND_TRUTH_DIR = 'ground_truth/ground_truth_'

def pixel_wise_accuracy(img, ground_truth):
    img = img_as_float(img)
    ground_truth = img_as_float(ground_truth)

    img_mse = mse(img, ground_truth)
    return img_mse

def ssim_measure(img, ground_truth):
    img = img_as_float(img)
    ground_truth = img_as_float(ground_truth)

    img_ssim = ssim(ground_truth, img, data_range=img.max()-img.min(), multichannel=True)
    return img_ssim 

def mse(x, y): 
    return np.linalg.norm(x-y)

def process_metrics(images, types):
    MSE = dict()
    SSIM = dict() 

    for t in types:
        if t == '': 
            MSE['baseline'] = list() 
            SSIM['baseline'] = list() 
        else: 
            MSE[t[0:-1]] = list() 
            SSIM[t[0:-1]] = list() 

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

            pixel_acc = pixel_wise_accuracy(img, ground_truth)
            ssim_val = ssim_measure(img, ground_truth)

            if t == '':
                MSE['baseline'].append(pixel_acc)
                SSIM['baseline'].append(ssim_val)
            else:
                MSE[t[0:-1]].append(pixel_acc)
                SSIM[t[0:-1]].append(ssim_val)

        print ('finished ' + i[0:-4])

    return MSE, SSIM 

def getMeanStd(d, types):
    avg, sd = list(), list()
    for t in types: 
        t = 'baseline' if t == '' else t[0:-1]

        print (t, round(np.mean(d[t]), 2), round(np.std(d[t]), 2))
        avg.append(round(np.mean(d[t]), 2))
        sd.append(round(np.std(d[t]), 2))

    return avg, sd 

def plotErrorStd(avg, sd, ylabel, title): 
    tick_name = ['base', 'blur', 'db', 'dk', 'kernel', 
                'kd', 'pd', 'c50', 'c100']

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

    TYPES = ['', 'new_', 'darken_', 'darken_kernel_', 'kernel_', 'kernel_darken_', 
        'pure_darken_', 'contrast_50_', 'contrast_100_']

    MSE, SSIM = process_metrics(IMAGES, TYPES)

    print ('--- MSE ---')
    mse_avg, mse_sd = getMeanStd(MSE, TYPES)

    print ('--- SSIM ---')
    ssim_avg, ssim_sd = getMeanStd(SSIM, TYPES)

    ylabel = 'MSE'
    title = 'Mean Squared Error vs. Techniques'
    plotErrorStd(mse_avg, mse_sd, ylabel, title)

    ylabel = 'SSIM'
    title = 'Structural Similarity Index Measure vs. Techniques'
    plotErrorStd(ssim_avg, ssim_sd, ylabel, title)


