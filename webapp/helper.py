import numpy as np

def compare_images(image_0, image_1):

    err = np.sum(abs(image_0.astype('float')-image_1.astype('float')))
    err /= float(image_0.shape[0] * image_1.shape[1])

    return err