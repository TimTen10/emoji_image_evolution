"""
Basic idea:
Recreate an image by approximating it using scaled and rotates components as building blocks.
The "emoji image" will be built one emoji after another using a genetic algorithm like approach.
All components found in the "components" folder can be used as parents / mutations.

Steps:
1. Load original image and turn it into height x width x colors matrix (colors = 4 if mode RGBA)
2.1 Create new (empty) image with the same width and height
2.2 Put one of the components (transformed) onto it
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

import glob
import random
from typing import Callable


def load_original(filepath: str) -> np.ndarray:
    original_image = Image.open(filepath).convert('RGBA')
    original_pixel = np.array(original_image, dtype=np.ubyte)  # Creates numpy array of pixels
    return original_pixel


def load_components(folderpath: str) -> list[Image]:
    """
    This method loads all '.png' as jpeg Images into a list.

    :param folderpath:
    :return: List of PIL.Images in RGB mode
    """
    return [Image.open(component) for component in glob.iglob(f"{folderpath}/*.png")]


def initialize_population(*, image_size: tuple[int, int], population_size: int) -> list[np.ndarray]:
    """
    Erzeugt die initiale Population (Liste) aus schwarzen Bildern (genauer gesagt Numpy Arrays mit nur 0en).

    :param image_size: Tupel aus Höhe x Breite
    :param population_size: Anzahl der Elemente in der Population
    :return: Liste aus 0er Arrays in der Größe des original Bildes (entspricht komplett schwarzen Bildern)
    """
    return [np.zeros(image_size, dtype=np.ubyte) for _ in range(population_size)]


def transform(base_image: np.ndarray, component: Image) -> np.ndarray:
    # Rotate:
    component = component.rotate(angle=random.randrange(360))

    # Scale:
    scaling_factor = random.randint(10, 1000) / 100
    new_dimensions = (int(component.width * scaling_factor), int(component.height * scaling_factor))
    component = component.resize(new_dimensions)

    # Position:
    height, width, _ = base_image.shape
    # Create
    canvas = np.zeros((height + component.height, width + component.width, 4), dtype=np.ubyte)
    x_pos = random.randint(0, width)
    y_pos = random.randint(0, height)
    canvas[y_pos:y_pos+component.height, x_pos:x_pos+component.width] = component
    # Cut into shape
    canvas = canvas[component.height // 2:-component.height // 2, component.width // 2:-component.width // 2]
    canvas_image = Image.fromarray(canvas)
    transformed_image = Image.fromarray(base_image)
    transformed_image.paste(canvas_image, (0, 0), canvas_image)
    return np.array(transformed_image, dtype=np.ubyte)


def score(original_image_pixels: np.ndarray, approach_image_pixels: np.ndarray) -> int:
    """
    Returns the sum of absolute pixel values attained by subtracting the approach image from the original.

    :param original_image_pixels: Original image pixel values
    :param approach_image_pixels: Approach image pixel values
    :return: score of "similarity" between pixel values
    """
    return np.sum(np.abs(original_image_pixels - approach_image_pixels)).item()


def evolve(population: list[np.ndarray],
           original: np.ndarray,
           components: list[Image],
           scoring_function: Callable[[np.ndarray, np.ndarray], float]) -> [np.ndarray, list[np.ndarray]]:
    scored = []
    for individual in population:
        scored.append((individual, scoring_function(original, individual)))

    scored.sort(key=lambda x: x[1])  # Sort scored individuals, best (lowest) scores at the front
    parent = scored[0][0]

    next_gen = []
    for comp in components:
        for _ in range(len(population) // len(components)):
            next_gen.append(transform(parent, comp))

    return parent, next_gen


def run():
    # TODO: Filenames via sysargs
    orig_name = "pokemon_ending_screen"
    orig_pix = load_original(f"original/{orig_name}.jpg")

    components = load_components("components")

    # Evolution
    epochs = 1000
    population_size = 100
    population = initialize_population(image_size=orig_pix.shape,
                                       population_size=population_size)

    for e in range(epochs + 1):
        best_image, population = evolve(population, orig_pix, components, score)
        if e % 2 == 0:
            best_image = Image.fromarray(best_image)
            best_image.save(f"output_images/{orig_name}_{e}.png")
        print(f"Done with Epoch {e}")


def main():
    im = Image.open("emojis/watermelon.png")
    print(im.format, im.size, im.mode)

    im_pix = np.array(im)
    print(im_pix.shape)

    new_image = np.zeros(im_pix.shape)
    print(new_image.shape)


if __name__ == '__main__':
    run()
