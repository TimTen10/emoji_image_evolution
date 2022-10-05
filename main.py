import PIL
import numpy as np
from PIL import Image

import argparse
import glob
import random
from pathlib import Path
from typing import Callable

from util import timed_func


def load_original() -> np.ndarray:
    """
    Loads the first image in the original folder as the original image.

    :return: Numpy array of the pixel values of the selected image.
    """
    try:
        original_image = Image.open(glob.glob("original/*")[0]).convert('RGBA')
    except (IndexError, PIL.UnidentifiedImageError):
        raise FileNotFoundError("There exists no .jpg / .jpeg / .png file in the 'original' folder") from None

    # Fill transparent spots with some color
    image_with_background = Image.new("RGBA", original_image.size, "WHITE")
    image_with_background.paste(original_image, (0, 0), original_image)

    original_pixel = np.array(image_with_background, dtype=np.ubyte)  # Creates numpy array of pixels
    return original_pixel


def load_components() -> list[Image]:
    """
    This method loads all '.png' PIL.Images into a list.

    :return: List of PIL.Images
    """
    return [Image.open(component) for component in glob.iglob(f"components/*.png")]


def initialize_population(*, image_size: tuple[int, int], population_size: int) -> list[np.ndarray]:
    """
    Erzeugt die initiale Population (Liste) aus schwarzen Bildern (genauer gesagt Numpy Arrays mit nur 0en).

    :param image_size: Tupel aus Höhe x Breite
    :param population_size: Anzahl der Elemente in der Population
    :return: Liste aus 0er Arrays in der Größe des original Bildes (entspricht komplett schwarzen Bildern)
    """
    return [np.zeros(image_size, dtype=np.ubyte) for _ in range(population_size)]


# @timed_func
def transform(base_image: np.ndarray, component: Image) -> np.ndarray:
    # Rotate:
    component = component.rotate(angle=random.randrange(360), expand=True)

    # Scale:
    scaling_factor = random.randint(10, 500) / 100
    new_dimensions = (int(component.width * scaling_factor), int(component.height * scaling_factor))
    component = component.resize(new_dimensions)

    # Position:
    height, width, _ = base_image.shape
    # Create
    x_pos = random.randint(0, width) - component.width // 2
    y_pos = random.randint(0, height) - component.height // 2
    transformed_image = Image.fromarray(base_image)
    transformed_image.paste(component, (x_pos, y_pos), component)
    return np.array(transformed_image, dtype=np.ubyte)


def score(original_image_pixels: np.ndarray, approach_image_pixels: np.ndarray) -> int:
    """
    Returns the sum of absolute pixel values attained by subtracting the approach image from the original.

    :param original_image_pixels: Original image pixel values.
    :param approach_image_pixels: Approach image pixel values.
    :return: score of "similarity" between pixel values.
    """
    return np.sum(np.abs(original_image_pixels - approach_image_pixels)).item()


@timed_func
def evolve(population: list[np.ndarray],
           original: np.ndarray,
           components: list[Image],
           scoring_function: Callable[[np.ndarray, np.ndarray], float]) -> [np.ndarray, list[np.ndarray]]:
    """
    Evolves a population of candidates by first scoring them and then merging transformed components onto copies of the
    best performing individual.

    :param population: List of candidate solutions.
    :param original: Original image to be approached.
    :param components: Building blocks for recreating the original image.
    :param scoring_function: Function for comparing how close the recreation is to the original.
    :return: Tuple of best performing individual and next generation population.
    """
    # TODO: Reintroduce having len(population) // 10 parents? Will slow down method unless we cut mutations per indiv
    parent = (population[0], scoring_function(original, population[0]))
    for individual in population[1:]:
        indiv_score = scoring_function(original, individual)
        if indiv_score < parent[1]:
            parent = (individual, indiv_score)

    parent = parent[0]
    next_gen = []
    for comp in components:
        for _ in range(len(population) // len(components)):
            next_gen.append(transform(parent, comp))

    return parent, next_gen


def run_evolution(run_name: str,
                  epochs: int,
                  population_size: int):
    # Load original image, components and initialize population
    orig_pix = load_original()
    components = load_components()
    population = initialize_population(image_size=orig_pix.shape,
                                       population_size=population_size)

    # Create output directory
    output_path = f"output_images/{run_name}_{epochs}_{population_size}"
    try:
        Path(output_path).mkdir(parents=True)
    except FileExistsError:
        pass

    # Evolution time
    for e in range(epochs + 1):
        best_image, population = evolve(population, orig_pix, components, score)
        if e % 20 == 0 and e != 0:
            best_image = Image.fromarray(best_image)
            best_image.save(f"{output_path}/recreation_epoch_{e}.png")
        print(f"Done with Epoch {e}")


def main():
    # Run some basic assertion tests
    im = np.full((123, 123, 4), 122, dtype=np.ubyte)

    assert score(im, im) == 0, "Score function should return 0 for same image"
    assert score(
        np.array([1, 2, 3, 4]),
        np.array([0, 0, 0, 0])
    ) == 10, "Score function returns false value"

    components = load_components()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recreate an image.")
    parser.add_argument("run_name", help="Name for output folder", type=str)
    parser.add_argument("epochs", help="Number of epochs to run the algorithm for", type=int)
    parser.add_argument("population_size", help="Number of individuals per epoch", type=int)
    args = parser.parse_args()

    run_evolution(run_name=args.run_name,
                  epochs=args.epochs,
                  population_size=args.population_size)
