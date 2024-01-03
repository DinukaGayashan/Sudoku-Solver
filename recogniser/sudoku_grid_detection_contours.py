import os

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from recogniser.cnn import SudokuNet

model = load_model('../Sudoku-Solver/mnist.h5')


def recognize_digit(image):
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    final_number = ''
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # make a rectangle box around each curve
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = th[y:y + h, x:x + w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0

        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)

        final_number += str(final_pred)

        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    return final_number


def run_recognizer(image):
    sudokunet = SudokuNet()

    file_path = "files\model\saved_model.pb"
    if not os.path.exists(file_path):
        sudokunet.train_model()

    image = cv2.bitwise_not(image)
    (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    digit = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]

    digit = cv2.resize(digit, (28, 28))
    cv2.imwrite('test.png', digit)
    # print(digit)
    digit = digit / 255.0

    img = img_to_array(digit)
    img = np.expand_dims(img, axis=0)
    return sudokunet.predict(img)


def read_sudoku_grid(result, sudoku_mode, matrix):
    print('selected mode:', sudoku_mode)

    digits = []
    for i in range(sudoku_mode):
        for j in range(sudoku_mode):
            height = result.shape[0]
            width = result.shape[1]

            region_of_interest = result[round((i * height / sudoku_mode)):round(((i + 1) * height / sudoku_mode)),
                                 round((j * width / sudoku_mode)):round(((j + 1) * width / sudoku_mode))]

            square = cv2.resize(region_of_interest, (160, 160))

            gray_scaled = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_scaled, (3, 3), 0)
            _, region_of_interest = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(region_of_interest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(result, largest_contour, -1, (0, 255, 0), 3)

            x, y, w, h = cv2.boundingRect(largest_contour)

            to_model = region_of_interest[y:y + h, x:x + w]

            digits.append(run_recognizer(to_model))

    print(digits)
    save_puzzle(sudoku_mode, digits)


def find_sudoku_grid(image):
    sudoku_area = [1 / 81, 1 / 256]
    puzzle_corners = []

    original_image = image.copy()

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) >= 4:
        sorted_points = sorted(approx, key=lambda x: x[0][1])
        # print('sorted points: ',sorted_points)

        if len(puzzle_corners) < 4:
            puzzle_corners.extend([None] * (4 - len(puzzle_corners)))

        if (sorted_points[0][0][0] < sorted_points[1][0][0]):
            puzzle_corners[0] = sorted_points[0].tolist()[0]
            puzzle_corners[1] = sorted_points[1].tolist()[0]
        else:
            puzzle_corners[0] = sorted_points[1].tolist()[0]
            puzzle_corners[1] = sorted_points[0].tolist()[0]

        if (sorted_points[-1][0][0] < sorted_points[-2][0][0]):
            puzzle_corners[2] = sorted_points[-1].tolist()[0]
            puzzle_corners[3] = sorted_points[-2].tolist()[0]
        else:
            puzzle_corners[2] = sorted_points[-2].tolist()[0]
            puzzle_corners[3] = sorted_points[-1].tolist()[0]

    image_height = original_image.shape[0]
    image_width = original_image.shape[1]
    homography_matrix, _ = cv2.findHomography(
        np.array([puzzle_corners[0], puzzle_corners[1], puzzle_corners[2], puzzle_corners[3]]),
        np.array([[0, 0], [image_width, 0], [0, image_height],
                  [image_width, image_height]]))

    warped_image = cv2.warpPerspective(original_image, homography_matrix, (image_width, image_height))
    warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    warped_blur = cv2.GaussianBlur(warped_gray, (3, 3), 0)
    warped_otsued = cv2.threshold(warped_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    warped_eroded = cv2.erode(warped_otsued, np.ones((3, 3), np.uint8), iterations=1)
    warped_edges = cv2.Canny(warped_eroded, 50, 150, apertureSize=3)

    save_warped_image(warped_image)
    # cv2.imshow('canny',warped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    warped_contours, _ = cv2.findContours(warped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(warped_image, warped_contours, -1, (0, 255, 0), 3)

    minimum_Area = original_image.shape[0] * original_image.shape[1]

    min_area_contour = None
    for contour in warped_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area < minimum_Area:
                minimum_Area = area
                min_area_contour = contour

    # print('minimum area cotour:',min_area_contour)   
    cv2.drawContours(warped_image, min_area_contour, -1, (0, 0, 255), 5)

    # cv2.imshow('min contour',warped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    minArea_precentage = minimum_Area / (original_image.shape[0] * original_image.shape[1])

    # print('minArea_precentage',minArea_precentage)

    if abs(minArea_precentage - sudoku_area[0]) < abs(minArea_precentage - sudoku_area[1]):
        sudoku_mode = 9
    else:
        sudoku_mode = 16

    # print('sudoku mode',sudoku_mode)

    matrix = [[0 for _ in range(sudoku_mode)] for _ in range(sudoku_mode)]

    read_sudoku_grid(warped_image, sudoku_mode, matrix)


def save_warped_image(image):
    height, width, _ = image.shape
    square_size = max(height, width)
    square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    scale_x = square_size / width
    scale_y = square_size / height

    stretched_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)

    x_offset = (square_size - stretched_image.shape[1]) // 2
    y_offset = (square_size - stretched_image.shape[0]) // 2

    square_image[y_offset:y_offset + stretched_image.shape[0],
    x_offset:x_offset + stretched_image.shape[1]] = stretched_image
    resized_image = cv2.resize(square_image, (960, 960))

    cv2.imwrite("files/puzzle.jpg", resized_image)


def save_puzzle(size, digits):
    puzzle = [[0] * size for _ in range(size)]

    for i in range(size):
        for j in range(size):
            puzzle[i][j] = digits[i * size + j]

    with open("files/puzzle.txt", 'w') as file:
        for row in puzzle:
            line = ' '.join(map(str, row)) + '\n'
            file.write(line)
