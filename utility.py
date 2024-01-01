import subprocess
import os
import cv2
from recogniser.sudoku_grid_detection_contours import find_sudoku_grid

def run_solver(input_filename):
    solver_name="solver/sudoku_solver"
    cpp_source = os.path.abspath(solver_name+".cpp")
    compiled_executable = os.path.abspath(solver_name)

    compile_command = ['g++', '-o', compiled_executable, cpp_source]
    subprocess.run(compile_command, check=True)

    run_command = [compiled_executable, input_filename]
    process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()

    if process.returncode == 0:
        # print(stdout.decode('utf-8'))
        return True

    if stderr:
        print(stderr.decode('utf-8'))
        return False
    


def get_solved_image(image,original_values,solved_values):
    sudoku_length=len(original_values)
    
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
            if original_values[i][j]==0 and solved_values[i][j] != 0:
                x_pos = j * cell_size + cell_size // 3
                y_pos = i * cell_size + 2 * cell_size // 3

                cell_value = str(solved_values[i][j])
                cv2.putText(image, cell_value, (x_pos, y_pos),
                            font, font_scale, font_color, font_thickness, line_type)

    return image

def extract_sudoku():
    find_sudoku_grid('files/sudoku.jpg')
    

