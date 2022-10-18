# 🎨 (Emoji) Image Evolution 🎨

This project is my reaction to a YouTube video in which the creator recreated images out of building blocks from a smartphone game.
In my version, called emoji image evolution, I do the same, recreating images, but with emojis instead of game blocks (and even these emojis are substitutable for any kind of .png image you want to use).

## How does it work?

The basic idea is:
Start with a blank image and put new components onto the image until it satisfatorily resembles the original image. For this process we use a form of genetic algorithm.

In more detail:
We start of with a population of candidate solutions to our recreation problem. At the start these solutions are the mentioned blank images. We proceed to mutate these individuals by putting a transformed (rotated, resized, randomly positioned) component onto each one of them. With the help of a scoring function, we decide which solution of the current population best solves our recreation problem. This individual is used as parent for the upcoming generation, meaning it is the basis onto which new transformed components are put. The whole process is repeated until either a fixed number of epochs are reached or the resulting image looks satisfactorily like the original input image.

A few observations:
* The first components put onto the blank image tend to be large ones, as they cover the most area and thus, with the basic score function, make the new image most similar to the original image.

* Have a wide range of color in the component images! The algorithm can only use the images that it gets provided, thus, if the closest color it gets to a yellow is red, it will substitute said yellow with red.

## Recreate your own images (Work in progress):

To recreate your own images out of your very own building blocks follow these 4 steps:
1. Clone this repository.
2. Create python environment according to the `requirements.txt` file.
3. In the same directory as the `main.py` file create the empty folders `components` and `original`.
4. Put the image you want recreated into the `original` folder (.jpeg or .png).
5. Put the components you want your image to be recreated from into the `components` folder (.png only, as non square images / images with transparent parts make for better shaping).
6. Run the `main.py` file with values for `epochs` (e.g. 1000) and `population_size` (e.g. 100).
7. Wait for images to be created in the `output_images` folder...
8. Enjoy!

## What to look out for!

* Algorithm still takes quite some time
