# Sudoku Solver

This application automatically recognizes Sudoku(9x9) and Hexadoku(16x16) puzzles and solves them.


## Overview

First, the input image taken from the file or camera feed is processed and the input digits are recognized. Afterwards the puzzle is being solved and the results are shown on the processed input image.

![Demo](demo.gif)


## Setup

1. Make sure Python3.10 and g++ is installed.
2. Clone the repository.
3. Install requirements.
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application.
    ```bash
    streamlit run main.py
    ```

Note: For solving Hexadoku puzzles, configure the [Google Cloud Vision API](https://cloud.google.com/vision/docs/ocr#vision_text_detection-python).
