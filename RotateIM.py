
def RotateFUNC(image):

    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import cv2 as cv
    import numpy as np
    import random

    """ 
    Function to rotate an image and return np array
    returns warped, warped_pil
    """

    img = np.array(image)
    (h, w) = img.shape[:2]  # getting the dimensions of the image
    center = (w // 2, h // 2)  # locating the center of the image
    angle = random.randint(0, 360) # random angle of rotation
    scale = .3

    M = cv.getRotationMatrix2D(center, angle, scale)  # generate the desired transformation matrix
    warped = cv.warpAffine(img, M, (w, h))  # transform np array

    plt.imshow(warped)
    warped_pil = Image.fromarray(warped)
    # warped_pil.show()
    return warped, warped_pil
