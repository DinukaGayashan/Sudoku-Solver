import subprocess
import os


def run_solver(input_file):
    cpp_source = os.path.abspath('solver/solver.cpp')
    compiled_executable = os.path.abspath('solver/solver')  # Specify the compiled executable path

    # Compile the C++ program
    compile_command = ['g++', '-o', compiled_executable, cpp_source]
    subprocess.run(compile_command, check=True)

    # Run the compiled C++ program
    run_command = [compiled_executable, input_file]
    process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and get the output
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print(stdout.decode('utf-8'))

    if stderr:
        print(stderr.decode('utf-8'))


if __name__ == "__main__":
    input_file = os.path.abspath('files/puzzle.txt')
    run_solver(input_file)
