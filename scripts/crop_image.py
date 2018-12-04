from PIL import Image
import os.path 

IMG_DIR = '../images/'
PREDICTED_DIR = 'predicted/'
CROPPED_DIR = 'cropped/'

TYPES = ['new_', 'darken_', 'darken_kernel_', 'kernel_', 'kernel_darken_', 
		'pure_darken_', 'contrast_50_', 'contrast_100_']
IMAGES = ['bedroom.png', 'dentist_office.png', 'elevator.png', 'ice_skating.png',
		'lake.png', 'lot.png', 'movie_theater.png', 'office.png', 'room.png']

def cropImage(img):
	img = Image.open(img)
	w, h = img.size 
	cropped_img = img.crop((w/2, 0, w, h))
	return cropped_img

def cropAll():
	IMG_FOLDER = IMG_DIR + PREDICTED_DIR
	SAVED_FOLDER = IMG_DIR + CROPPED_DIR

	for i in IMAGES: 
		for t in TYPES:
			img = t + i
			img_path = IMG_FOLDER + img
			if not os.path.isfile(img_path): 
				print (t+i + ' does not exist!')
			else: 
				cropped_img = cropImage(img_path)
				cropped_img.save(SAVED_FOLDER + img)

		print ('finished ' + i[0:-4])

cropAll()