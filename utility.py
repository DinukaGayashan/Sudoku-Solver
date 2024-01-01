import subprocess
import os

def run_solver(input_file):
    solver_name="solver/sudoku_solver"
    cpp_source = os.path.abspath(solver_name+".cpp")
    compiled_executable = os.path.abspath(solver_name)

    compile_command = ['g++', '-o', compiled_executable, cpp_source]
    subprocess.run(compile_command, check=True)

    run_command = [compiled_executable, input_file]
    process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()

    # if process.returncode == 0:
    #     print(stdout.decode('utf-8'))

    # if stderr:
    #     print(stderr.decode('utf-8'))


def is_valid_sudoku(board,size) -> bool:
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
