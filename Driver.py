from PIL import Image, ImageDraw, ImageFont
import numpy as np

import RotateIM as Rt
import MarkIM as Mk

img_pil = Image.open("/Users/nathanaelseay/Documents/DL/Driver.png")

# Mark image
Marks = []

for count in range(2000):

    Mark, Mark_pil= Mk.MarkFUNC(img_pil) # Marking function
    Marks.append(Mark)
    Marked_rotated, Marked_rotated_pil= Rt.RotateFUNC(Mark_pil)

    name = '/Users/nathanaelseay/Documents/DL/training_data/marked/DL_marked_image' + str(count) + '.jpg'
    Marked_rotated_pil.save(name)
# Rotate image:

for count in range(2000):

    rotated_clean, rotated_pil_clean= Rt.RotateFUNC(img_pil) # Rotation function
    name = '/Users/nathanaelseay/Documents/DL/training_data/clean/DL_clean_image' + str(count) + '.jpg'
    rotated_pil_clean.save(name)

# for count in range(Marks.__len__()):





