import os

import streamlit as st

import utility

if __name__ == "__main__":
    st.title("Sudoku Solver")
    st.divider()
    left_col, right_col = st.columns([1, 2])
    with left_col.container():
        input_method = st.radio("Select the input method", ["Input file", "Camera feed"])
    with right_col.container():
        if input_method == "Input file":
            file = st.file_uploader("Select Sudoku image", type=["jpg", "png"])
        if input_method == "Camera feed":
            file = st.camera_input("Take Sudoku image")
    st.title("")
    solve = st.button("Solve", disabled=False if file else True)
    st.divider()
    if solve:
        with st.spinner("Please wait while puzzle is solving."):
            utility.extract_sudoku(file)
            utility.get_puzzle("files/puzzle.jpg",9)
            valid_sudoku = utility.is_valid_sudoku("files/puzzle.txt")
            if utility.is_valid_sudoku('files/puzzle.txt'):

                input_file = os.path.abspath("files/puzzle.txt")
                solved = utility.run_solver(input_file)

                if solved:
                    with open("files/puzzle_output.txt", "r") as solved_file, open(
                        "files/puzzle.txt", "r"
                    ) as original_file:
                        solved_values = [list(map(int, line.strip().split())) for line in solved_file]
                        original_values = [list(map(int, line.strip().split())) for line in original_file]

                        solved_image = utility.get_solved_image("files/puzzle.jpg", original_values, solved_values)
                        st.image(solved_image, caption="Solved Sudoku puzzle")
                else:
                    st.error("Issue in solving Sudoku.")
            else:
                st.error("Sudoku is invaid.")
