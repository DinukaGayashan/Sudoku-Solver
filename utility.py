import os
import subprocess

import cv2
import imutils
import numpy as np
import pytesseract
from keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
# from keras.models import load_model

from recogniser.cnn import SudokuNet
from recogniser.recogniser import image_processing
from recogniser.sudoku_grid_detection_contours import find_sudoku_grid

# classifier=load_model("files\digit_model.h5")

# def add_pading(image):
#     top_pad = 10
#     bottom_pad = 10
#     left_pad = 10
#     right_pad = 10
#     return cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))



def extract_sudoku(file, original_image, processed_image,size_file):
    with open(original_image, "wb") as f:
        f.write(file.getbuffer())
    size=find_sudoku_grid(original_image)
    with open(size_file, "w") as file:
        file.write(str(size))
    
    image_processing(original_image, processed_image,size)
    # find_sudoku_grid(image)


pred_count = 0


def apply_raw_cell(cell):
    img = cell.copy()
    mask = np.ones_like(img)
    thicknes = 10
    mask[thicknes:-thicknes, thicknes:-thicknes] = 0
    img[mask.astype(bool)] = 0
    return img


def extract_digit(cell, i, j, sudokunet):
    # cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    img = apply_raw_cell(cell)
    thresh = cv2.threshold(
        cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    # cv2.imwrite(f'to_model{i}{j}.png',img)
    cnts = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return 0

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(cell.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = cell.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    if percentFilled < 0.03:
        return 0

    digit = cv2.bitwise_and(cell, cell, mask=mask)

    # Create a mask to select the region without the outer pixels

    padding_top = 4
    padding_bottom = 4
    padding_left = 4
    padding_right = 4

    border_type = cv2.BORDER_CONSTANT
    padding_color = (0, 0, 0)

    digit = cv2.copyMakeBorder(
        digit, padding_top, padding_bottom, padding_left, padding_right, border_type, value=padding_color
    )

    mask = np.ones_like(digit)
    thicknes = 12
    mask[thicknes:-thicknes, thicknes:-thicknes] = 0

    # Set the outer pixels to black
    digit[mask.astype(bool)] = 0

    roi = cv2.resize(digit, (28, 28))

    try:
        tess_pred = int(
            pytesseract.image_to_string(
                digit, config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789 outputbase digits"
            ).strip()
        )
        print(tess_pred)
    except Exception as e:
        tess_pred = 0

    # cv2.imwrite(f"test{i}{j}.png", roi)
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    pred, confident = sudokunet.predict_(roi)
    # predictions = classifier.predict(roi)
    # predicted_classes = np.argmax(predictions, axis=1)

    # print(predicted_classes)
    # return predicted_classes[0]
    global pred_count
    pred_count += 1

    if confident * 100 < 60:
        pred = tess_pred
        if pred < 1 or pred > 9:
            roi2 = cv2.resize(img, (28, 28))
            roi2 = img_to_array(roi2)
            roi2 = np.expand_dims(roi2, axis=0)
            pred2, confident2 = sudokunet.predict_(roi2)
            if confident2 * 100 > 60:
                pred = pred2
            else:
                pred = 0

    print(f"[{i}][{j}] = {pred} - {confident}%")
    print(pred_count)
    return pred


def get_puzzle(filename, extracted_puzzle, size_file):
    with open(size_file, "r") as file:
        sudoku_mode = int(file.readline())

    image = cv2.imread(filename)
    # num_rows = puzzle_size
    # num_cols = puzzle_size
    
    height, width = image.shape[:2]
    square_height = height // sudoku_mode
    square_width = width // sudoku_mode

    puzzle = []

    sudokunet = SudokuNet()

    for i in range(sudoku_mode):
        row = []
        for j in range(sudoku_mode):

            height=image.shape[0]
            width=image.shape[1]
    
            region_of_interest = image[round((i * height / sudoku_mode)):round(((i + 1) *height / sudoku_mode)),
            round((j * width / sudoku_mode)):round(((j + 1) * width / sudoku_mode))]

            square = cv2.resize(region_of_interest, (160, 160))

            gray_scaled = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_scaled, (3, 3), 0)
            _, region_of_interest = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(region_of_interest , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(square,contours, -1, (0, 255, 0), 3)

            largest_contour = max(contours, key=cv2.contourArea)

            # cv2.imshow('largest contour',square)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            x, y, w, h = cv2.boundingRect(largest_contour)

            to_model  = region_of_interest[y:y + h, x:x + w]

            # cropped= cv2.morphologyEx(region_of_interest, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            # return square
            # cv2.imwrite(f'to_model{i}{j}.png',to_model)
            number = extract_digit(to_model, i, j, sudokunet)
            row.append(0 if number is None else number)
        puzzle.append(row)

    # for i in range(sudoku_mode):
    #     row = []
    #     for j in range(sudoku_mode):
    #         y1 = i * square_height
    #         y2 = (i + 1) * square_height
    #         x1 = j * square_width
    #         x2 = (j + 1) * square_width
    #         square_part = image[y1:y2, x1:x2]
    #         cv2.imshow('largest contour',square_part)
    #         cv2.waitKey(0)
    #         s = extract_digit(square_part, i, j, sudokunet)
    #         number = s
    #         row.append(0 if number is None else number)
    #     puzzle.append(row)

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
        run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()

    if process.returncode == 0:
        # print(stdout.decode('utf-8'))
        return True

    if stderr:
        print(stderr.decode('utf-8'))
        return False


def get_solved_image(image, original_puzzle, solved_puzzle):
    with open(solved_puzzle, "r") as solved_file, open(
        original_puzzle, "r"
    ) as original_file:
        solved_values = [list(map(int, line.strip().split()))
                         for line in solved_file]
        original_values = [list(map(int, line.strip().split()))
                           for line in original_file]

    sudoku_length = len(original_values)

    image = cv2.imread(image)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    font_thickness = 2
    font_color = (100, 180, 100)
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
            if (curr in rows[i]) or (curr in cols[j]) or (curr in block[i // sqrt][j // sqrt]):
                return False
            rows[i].add(curr)
            cols[j].add(curr)
            block[i // sqrt][j // sqrt].add(curr)
    return True
