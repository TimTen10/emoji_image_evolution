"""
Basic idea:
Recreate an image by approximating it using scaled and rotates building_blocks as building blocks.
The "emoji image" will be built one emoji after another using a genetic algorithm like approach.
All building_blocks found in the "building_blocks" folder can be used as parents / mutations.

Steps:
1. Load original image and turn it into height x width x colors matrix (colors = 3 if mode RGB, = 4 if mode RGBA)
2.1 Create new (empty) image with the same width and height
2.2 Put one of the building_blocks (transformed) onto it
    - possible transformations:
        - rotate
        - size
        - position
        -> randomly create transformations
3. Compare the new image and the original
    - Repeat for every version of the new image
4. Choose best one
    - Need function to determine the "best" one (pixel value diff.?)
5. Repeat until new image is close enough to original image
"""

import numpy as np
from PIL import Image

from typing import Callable


def load_original(filepath: str) -> np.ndarray:
    original_image = Image.open(filepath)
    original_pixel = np.array(original_image)  # Creates numpy array of pixels
    return original_pixel


def initialize_population(*, image_size: tuple[int, int], population_size: int) -> list[np.ndarray]:
    return [np.zeros(image_size) for _ in range(population_size)]


def transform(to_add: np.ndarray) -> np.ndarray:
    # TODO: Might want to have PIL.Image as input instead of np.ndarray for rotation and scaling purpose
    # Position can not really be transformed in here
    pass


def score(original_image_pixels: np.ndarray, approach_image_pixels: np.ndarray) -> float:
    pass


def evolve(population: list[np.ndarray],
           scoring_function: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    pass


def main():
    im = Image.open("emojis/watermelon.png")
    print(im.format, im.size, im.mode)

    im_pix = np.array(im)
    print(im_pix.shape)

    new_image = np.zeros(im_pix.shape)
    print(new_image.shape)


if __name__ == '__main__':
    main()
