def MarkFUNC(image):

    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import cv2 as cv
    import numpy as np
    import random

    """ 
    Function generates a random redaction on image
    """

    img = np.array(image)
    (h, w) = img.shape[:2]  # getting the dimensions of the image
    x1 = np.random.randint(0, w)
    y1 = np.random.randint(0, h)
    x2 = np.random.randint(0, w)
    y2 = np.random.randint(0, h)


    warped = cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)


    plt.imshow(warped)
    warped_pil = Image.fromarray(warped)
    # warped_pil.show()
    return warped, warped_pil
