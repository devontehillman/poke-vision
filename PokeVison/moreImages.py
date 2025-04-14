'''
Authors: Evan Gronewold, Devonte Hillman, Svens Daukss
Overview:
    This script is designed to download images from the internet using the 
    `simple_image_download` library. It provides a function `download_images` 
    that takes a search query and a limit on the number of images to download. 
    The script demonstrates downloading 100 images for each specified Pokémon 
    card class.
Resources Used:
    `simple_image_download` library: A Python library for downloading images 
    from the internet based on a search query.
Functions:
    download_images(query, limit=50): Downloads images based on the provided 
    search query and limit.
Usage:
    The script is pre-configured to download images for specific Pokémon card 
    classes such as "Bulbasaur card", "Squirtle card", etc., with a limit of 
    100 images per class.
'''

from simple_image_download import simple_image_download as simp

def download_images(query, limit=50):
    response = simp.simple_image_download
    response().download(query, limit=limit)
    
#down loading 100 images from each class
download_images("Bulbasaur card", 100)
download_images("Squirtle card", 100)
download_images("Cyndaquil card", 100)
download_images("Totodile card", 100)
download_images("Chikorita card", 100)