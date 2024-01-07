import os

import streamlit as st

import utility


if __name__ == "__main__":
    st.title("Sudoku Solver")
    st.divider()

    left_col, right_col = st.columns([1, 2])
    with left_col.container():
        input_method = st.radio("Select the input method", [
                                "Input file", "Camera feed"])
    with right_col.container():
        if input_method == "Input file":
            file = st.file_uploader("Select Sudoku image", type=["jpg", "png"])
            # if file:
            #     file = utility.add_pading(file.getbuffer())
        if input_method == "Camera feed":
            file = st.camera_input("Take Sudoku image")
    st.title("")

    solve = st.button("Solve", disabled=False if file else True)
    st.divider()
    if solve:
        with st.spinner("Please wait while puzzle is solving."):
            original_image = os.path.join("files", "sudoku.jpg")
            processed_image = os.path.join("files", "puzzle.jpg")
            image_to_model = os.path.join("files", "to_model.jpg")
            extracted_puzzle = os.path.join("files", "puzzle.txt")
            solved_puzzle = os.path.join("files", "puzzle_output.txt")
            size_file = os.path.join("files", "puzzle_size")
            solver_name = os.path.join("solver", "sudoku_solver")

            utility.extract_sudoku(file, original_image,
                                   processed_image,image_to_model,size_file)
            utility.get_puzzle(image_to_model, extracted_puzzle, size_file)
            if utility.is_valid_sudoku(extracted_puzzle):
                input_file = os.path.abspath(extracted_puzzle)
                if utility.run_solver(solver_name, input_file):
                    solved_image = utility.get_solved_image(
                        processed_image, extracted_puzzle, solved_puzzle)
                    st.image(solved_image, caption="Solved Sudoku puzzle")
                else:
                    st.error("Issue in solving Sudoku.")
            else:
                st.error("Sudoku is invaid. Try again.")

