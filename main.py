import os
import streamlit as st
from utility import run_solver


def gui():
    st.title('Sudoku Solver')
    st.divider()
    left_col, right_col = st.columns([1, 2])
    with left_col.container():
        input_method=st.radio("Select the input method",["Input file", "Camera feed"])
    with right_col.container():
        if input_method=="Input file":
            file=st.file_uploader("Select Sudoku image", type=["jpg", "png"])
        if input_method=="Camera feed":
            file=st.camera_input("Take Sudoku image")
    st.divider()


if __name__ == "__main__":
    gui()
    # input_file = os.path.abspath('files/puzzle.txt')
    # run_solver(input_file)
