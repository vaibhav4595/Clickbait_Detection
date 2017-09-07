import numpy as np
from scipy import misc
import tensorflow as tf
import cv2

# VGG 16 accepts RGB channel 0 to 1 (This tensorflow model).
def load_image_array(image_file):
    # img = misc.imread(image_file)
    img = cv2.imread(image_file)
    # GRAYSCALE
    if img is None:
        return np.zeros((224, 224, 3)).astype('float32')
    if len(img.shape) == 2:
	img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
	img_new[:,:,0] = img
	img_new[:,:,1] = img
	img_new[:,:,2] = img
	img = img_new

    img_resized = misc.imresize(img, (224, 224, 3))
    return (img_resized/255.0).astype('float32')
