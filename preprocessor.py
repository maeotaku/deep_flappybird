import numpy as np
from PIL import Image
from scipy.misc import imresize

class Preprocessor():

    @classmethod
    def save_img(cls, img, path):
        im = Image.fromarray(img)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save("your_file.jpeg")

    @classmethod
    def crop(cls, img, crop_size):
        return img[:,:crop_size]

    @classmethod
    def to_gray(cls, img):
        photo_data = img.mean(axis=-1)
        return photo_data

    @classmethod
    def to_binary(cls, img):
        #img[img <= 50] = 0
        #img[img > 50] = 255
        return img

    @classmethod
    def resize(cls, img, input_shape):
        return imresize(img, input_shape)#.reshape()

    @classmethod
    def normalize(cls, x):
        x_float = np.zeros(input_shape, dtype=np.float)
        x_float = x / 255.0
        return x_float

def pre_process(img, input_shape, crop_size):
    img = Preprocessor.crop(img, crop_size)
    img = Preprocessor.to_gray(img)
    img = Preprocessor.resize(img, input_shape)
    img = Preprocessor.to_binary(img)
    return img.reshape(input_shape)#.ravel()
