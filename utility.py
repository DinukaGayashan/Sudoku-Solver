import os
import subprocess

import cv2

from recogniser.sudoku_grid_detection_contours import find_sudoku_grid


def extract_sudoku(filename):
    image = cv2.imread(filename)
    find_sudoku_grid(image)


def run_solver(input_filename):
    solver_name = "solver/sudoku_solver"
    cpp_source = os.path.abspath(solver_name + ".cpp")
    compiled_executable = os.path.abspath(solver_name)

    compile_command = [
        "g++",
        "-std=c++17",
        "-o",
        compiled_executable,
        cpp_source,
    ]
    subprocess.run(compile_command, check=True)

    run_command = [compiled_executable, input_filename]
    subprocess.run(run_command)

    return True


def get_solved_image(image, original_values, solved_values):
    sudoku_length = len(original_values)

    image = cv2.imread(image)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2
    font_thickness = 2
    font_color = (120, 60, 120)
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
    with open(filename, 'r') as file:
        lines = file.readlines()

    board = [list(map(int, line.strip().split())) for line in lines]
    size=len(board[0])

    sqrt=int(size**0.5)
    rows = [set() for _ in range(size)]
    cols = [set() for _ in range(size)]
    block = [[set() for _ in range(sqrt)] for _ in range(sqrt)]

    for i in range(size):
        for j in range(size):
            curr = board[i][j]
            if curr not in range(0, size+1):
                return False
            if curr == 0:
                continue
            if (curr in rows[i]) or (curr in cols[j]) or (curr in block[i // sqrt][j // sqrt]):
                return False
            rows[i].add(curr)
            cols[j].add(curr)
            block[i // sqrt][j // sqrt].add(curr)
    return True