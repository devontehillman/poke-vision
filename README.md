# poke-vision

## Overview
**poke-vision** is a project focused on applying machine learning techniques to analyze and predict patterns in Pokémon data. This project is part of the coursework for CIS 378 01: Applied Machine Learning.

## Features
- Data analysis and visualization of Pokémon datasets.
- Implementation of machine learning models to predict Pokémon attributes.

## Prerequisites
- Python 3.8 or higher

## Installation
1. Clone the repository:
    ```terminal
    git clone https://github.com/yourusername/poke-vision.git
    ```
2. Navigate to the project directory:
    ```terminal
    cd poke-vision
    ```
3. Install the required dependencies:
    ```terminal 
    pip install -r requirements.txt
    ```

## Usage
1. Fetch images from Google by running the following program:
    ```terminal
    python moreImages.py
    ```
    - After fetching, clean the images by removing corrupted files and incorrect images for the class.
    - Organize the images into a directory structured as follows:
        ```
        pokemon_data/
            raw/
                Charmander/
                Bulbasaur/
                Squirtle/
                Cyndaquil/
                Totodile/
                Chikorita/
        ```

2. Train the model using the dataset:
    ```terminal
    python pokevision.py
    ```

3. Test the model with existing data without retraining:
    ```terminal
    python testVision.py
    ```

4. Run `oneImage.py` to pass a single image to the model for prediction:
    1. Create a directory named `oneImage`:
        ```terminal
        mkdir oneImage
        ```
    2. Add the image file you want to predict into the `oneImage` directory.
    3. Edit the `oneImage.py` file to load the specific image file for prediction.
    4. Execute the program:
        ```terminal
        python oneImage.py
        ```
