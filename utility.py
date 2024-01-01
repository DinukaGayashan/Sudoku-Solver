import subprocess
import os

def run_solver(input_file):
    cpp_source = os.path.abspath('solver/sudoku_solver.cpp')
    compiled_executable = os.path.abspath('solver/sudoku_solver')

    compile_command = ['g++', '-o', compiled_executable, cpp_source]
    subprocess.run(compile_command, check=True)

    run_command = [compiled_executable, input_file]
    process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(stdout.decode('utf-8'))

    if stderr:
        print(stderr.decode('utf-8'))
