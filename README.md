# poke-vision

## Overview
**poke-vision** is a project focused on applying machine learning techniques to analyze and predict patterns in Pokémon data. This project is part of the coursework for CIS 378 01: Applied Machine Learning.

## Features
- Data analysis and visualization of Pokémon datasets.
- Implementation of machine learning models to predict Pokémon attributes.
- Exploration of classification and regression techniques.

## Prerequisites
- Python 3.8 or higher
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/poke-vision.git
    ```
2. Navigate to the project directory:
    ```bash
    cd poke-vision
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Fetch images from Google by running the following script:
    ```bash
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
    ```bash
    python pokevision.py
    ```

3. Test the model with existing data without retraining:
    ```bash
    python testVision.py
    ```

4. Run `oneImage.py` to pass a single image to the model for prediction:
    1. Create a directory named `oneImage`:
        ```bash
        mkdir oneImage
        ```
    2. Add the image file you want to predict into the `oneImage` directory.
    3. Edit the `oneImage.py` script to load the specific image file for prediction.
    4. Execute the script:
        ```bash
        python oneImage.py
        ```

## Future Improvements

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
