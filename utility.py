import multiprocessing
import os
import subprocess

import cv2
import imutils
import numpy as np
from google.cloud import vision
from keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
from tensorflow.keras.models import load_model

from recogniser.sudoku_grid_detection import find_sudoku_grid
from recogniser.sudoku_grid_processor import process_image

model_path = os.path.join("files", "digit_model.h5")
sudokunet = load_model(model_path)


def extract_sudoku(file, original_image, processed_image, image_to_model, size_file):
    with open(original_image, "wb") as f:
        f.write(file.getbuffer())

    size = find_sudoku_grid(original_image, image_to_model, processed_image)
    with open(size_file, "w") as file:
        file.write(str(size))

    if size == 9:
        process_image(original_image, image_to_model, processed_image, size)


def detect_text(image):
    prediction = 0
    try:
        client = vision.ImageAnnotatorClient()

        _, processed_image_data = cv2.imencode(".png", np.array(image, dtype=np.uint8))
        content = processed_image_data.tobytes()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)

        if response.error.message:
            raise Exception(str(format(response.error.message)))

        texts = response.text_annotations
        size = len(texts)
        if size:
            prediction = int(texts[0].description.strip())

        return prediction

    except Exception as e:
        print(f"Error: {e}")
        raise Exception(e)


def apply_raw_cell(cell):
    img = cell.copy()
    mask = np.ones_like(img)
    thicknes = 10
    mask[thicknes:-thicknes, thicknes:-thicknes] = 0
    img[mask.astype(bool)] = 0
    return img


def get_digit(cell):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    cnts = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return 0

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(cell.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = cell.shape
    percent_filled = cv2.countNonZero(mask) / float(w * h)
    if percent_filled < 0.03:
        return 0

    digit = cv2.bitwise_and(cell, cell, mask=mask)
    padding = 4
    border_type = cv2.BORDER_CONSTANT
    padding_color = (0, 0, 0)
    digit = cv2.copyMakeBorder(
        digit,
        padding,
        padding,
        padding,
        padding,
        border_type,
        value=padding_color,
    )

    mask = np.ones_like(digit)
    thicknes = 12
    mask[thicknes:-thicknes, thicknes:-thicknes] = 0
    digit[mask.astype(bool)] = 0
    roi = cv2.resize(digit, (28, 28))
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    return sudokunet.predict(roi).argmax(axis=1)[0]


def extract_digit(sudoku_mode, cell):
    if sudoku_mode == 9:
        pred = get_digit(cell)
    else:
        img = apply_raw_cell(cell)
        pred = detect_text(img)
        if pred == 6 or pred == 9:
            cell = cv2.bitwise_not(cell)
            pred = get_digit(cell)

    return pred


def get_puzzle(filename, extracted_puzzle, size_file):
    with open(size_file, "r") as file:
        sudoku_mode = int(file.readline())
    image = cv2.imread(filename)

    multi_process_puzzle = []
    for i in range(sudoku_mode):
        row = []
        for j in range(sudoku_mode):
            height = image.shape[0]
            width = image.shape[1]

            region_of_interest = image[
                round((i * height / sudoku_mode)) : round(
                    ((i + 1) * height / sudoku_mode)
                ),
                round((j * width / sudoku_mode)) : round(
                    ((j + 1) * width / sudoku_mode)
                ),
            ]
            square = cv2.resize(region_of_interest, (160, 160))
            gray_scaled = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_scaled, (3, 3), 0)
            _, region_of_interest = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            contours, _ = cv2.findContours(
                region_of_interest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(square, contours, -1, (0, 255, 0), 3)

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            to_model = region_of_interest[y : y + h, x : x + w]

            multi_process_puzzle.append((sudoku_mode, to_model))
    with multiprocessing.Pool(processes=sudoku_mode) as pool:
        results = pool.starmap(extract_digit, multi_process_puzzle)

    puzzle = [results[i : i + sudoku_mode] for i in range(0, len(results), sudoku_mode)]
    with open(extracted_puzzle, "w") as file:
        for row in puzzle:
            file.write(" ".join(map(str, row)) + "\n")


def run_solver(solver_name, input_filename):
    cpp_source = os.path.abspath(solver_name + ".cpp")
    compiled_executable = os.path.abspath(solver_name)

    compile_command = [
        "g++",
        "-std=c++17",
        "-o",
        compiled_executable,
        cpp_source,
    ]
    subprocess.run(compile_command)

    run_command = [compiled_executable, input_filename]
    process = subprocess.Popen(
        run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    stdout, stderr = process.communicate()

    if process.returncode == 0:
        return True

    if stderr:
        print(stderr.decode("utf-8"))
        return False


def is_valid_sudoku(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    board = [list(map(int, line.strip().split())) for line in lines]
    size = len(board[0])

    sqrt = int(size**0.5)
    rows = [set() for _ in range(size)]
    cols = [set() for _ in range(size)]
    block = [[set() for _ in range(sqrt)] for _ in range(sqrt)]

    for i in range(size):
        for j in range(size):
            curr = board[i][j]
            if curr not in range(0, size + 1):
                return False
            if curr == 0:
                continue
            if (
                (curr in rows[i])
                or (curr in cols[j])
                or (curr in block[i // sqrt][j // sqrt])
            ):
                return False
            rows[i].add(curr)
            cols[j].add(curr)
            block[i // sqrt][j // sqrt].add(curr)
    return True


def get_solved_image(image, original_puzzle, solved_puzzle):
    with open(solved_puzzle, "r") as solved_file, open(
        original_puzzle, "r"
    ) as original_file:
        solved_values = [list(map(int, line.strip().split())) for line in solved_file]
        original_values = [
            list(map(int, line.strip().split())) for line in original_file
        ]

    sudoku_length = len(original_values)
    image = cv2.imread(image)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2 if sudoku_length == 9 else 1
    font_thickness = 2
    font_color = (60, 120, 60)
    line_type = cv2.LINE_AA

    height, _, _ = image.shape
    cell_size = height // sudoku_length

    for i in range(sudoku_length):
        for j in range(sudoku_length):
            if original_values[i][j] == 0:
                x_pos = j * cell_size + cell_size // 3
                y_pos = i * cell_size + 2 * cell_size // 3

                cell_value = str(solved_values[i][j])
                cv2.putText(
                    image,
                    cell_value,
                    (x_pos, y_pos),
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                    line_type,
                )
    return image
