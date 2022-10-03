"""
Basic idea:
Recreate an image by approximating it using scaled and rotates emojis as building blocks.
The "emoji image" will be built one emoji after another using a genetic algorithm like approach.
All emojis found in the "emojis" folder can be used as parents / mutations.

Steps:
1.) Load original image and turn it into height x width x colors matrix (colors = 3 if mode RGB, = 4 if mode RGBA)
2.1) Create new (empty) image with the same width and height
2.2) Put one of the emojis (transformed) onto it
    - possible transformations:
        - rotate
        - size
        - position
        -> randomly create transformations
3.) Compare the new image and the original
    - Repeat for every version of the new image
4.) Choose best one
    - Need function to determine the "best" one (pixel value diff.?)
5.) Repeat until new image is close enough to original image
"""

import numpy as np
from PIL import Image


def main():
    im = Image.open("emojis/watermelon.png")
    print(im.format, im.size, im.mode)

    im_pix = np.array(im)
    print(im_pix.shape)

    new_image = np.zeros(im_pix.shape)
    print(new_image.shape)


if __name__ == '__main__':
    main()
