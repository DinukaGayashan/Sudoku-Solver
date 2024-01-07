import cv2
import numpy as np


def read_sudoku_grid(result,sudoku_mode,matrix):
    # size=400
    for i in range(sudoku_mode):
        for j in range(sudoku_mode):

            height=result.shape[0]
            width=result.shape[1]
    
            region_of_interest = result[round((i * height / sudoku_mode)):round(((i + 1) *height / sudoku_mode)),
            round((j * width / sudoku_mode)):round(((j + 1) * width / sudoku_mode))]

            square = cv2.resize(region_of_interest, (160, 160))

            gray_scaled = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_scaled, (3, 3), 0)
            _, region_of_interest = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(region_of_interest , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(square,contours, -1, (0, 255, 0), 3)

            largest_contour = max(contours, key=cv2.contourArea)

            cv2.imshow('largest contour',square)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            x, y, w, h = cv2.boundingRect(largest_contour)

            to_model  = region_of_interest[y:y + h, x:x + w]

            cropped= cv2.morphologyEx(region_of_interest, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            #send to model


def add_pading(image):
    padding_top = 10
    padding_bottom = 10
    padding_left = 10
    padding_right = 10

    border_type = cv2.BORDER_CONSTANT
    padding_color = (255, 255, 255)

    image = cv2.copyMakeBorder(
        image, padding_top, padding_bottom, padding_left, padding_right, border_type, value=padding_color
    )
    return image

def find_sudoku_grid(original_image,image_to_model,processed_image):

    sudoku_area=[1/81,1/256]
    puzzle_corners=[]

    image = cv2.imread(original_image)
    image=add_pading(image)
    # cv2.imwrite(f'x.png',image)
    original_image = image.copy()

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # thershed =cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True) 

    if len(approx)>=4:
        sorted_points = sorted(approx, key=lambda x: x[0][1])

        if len(puzzle_corners) < 4:
            puzzle_corners.extend([None] * (4 - len(puzzle_corners)))

        if (sorted_points[0][0][0] < sorted_points[1][0][0]):
            puzzle_corners[0] = sorted_points[0].tolist()[0]
            puzzle_corners[1] = sorted_points[1].tolist()[0]
        else:
            puzzle_corners[0]= sorted_points[1].tolist()[0]
            puzzle_corners[1] = sorted_points[0].tolist()[0]

        if (sorted_points[-1][0][0] < sorted_points[-2][0][0]):
            puzzle_corners[2] = sorted_points[-1].tolist()[0]
            puzzle_corners[3]= sorted_points[-2].tolist()[0]
        else:
            puzzle_corners[2] = sorted_points[-2].tolist()[0]
            puzzle_corners[3] = sorted_points[-1].tolist()[0]

    
    image_height=original_image.shape[0]
    image_width=original_image.shape[1]
    homography_matrix, _ = cv2.findHomography(np.array([puzzle_corners[0], puzzle_corners[1], puzzle_corners[2],puzzle_corners[3]]),
                                            np.array([[0, 0], [ image_width, 0], [0,  image_height],
                                                        [ image_width, image_height]]))
    
    warped_image = cv2.warpPerspective( original_image, homography_matrix, (image_width, image_height))
    warped_image = cv2.resize(warped_image, (400, 400))
    warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    warped_blur= cv2.GaussianBlur(warped_gray, (3, 3), 0)
    warped_otsued = cv2.threshold(warped_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    warped_eroded = cv2.erode(warped_otsued, np.ones((3, 3), np.uint8), iterations=1)
    warped_edges = cv2.Canny(warped_eroded, 50, 150, apertureSize=3)

    warped_contours, _ = cv2.findContours(warped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minimum_area =  400*400
    area=minimum_area

    for contour in warped_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cell_area=cv2.contourArea(approx)


        if len(approx) == 4 and area//500 < cell_area< area:
            if cell_area>0 and cell_area<minimum_area:
                minimum_area = cell_area


    #set the default mode=9
    sudoku_mode=9

    area_proportion=area//minimum_area

    if area_proportion in range(6,12):
        sudoku_mode=9
    elif area_proportion in range(12,20):
        sudoku_mode=16
    elif area_proportion in range(80,256):
        sudoku_mode=9
    elif area_proportion>256:
        sudoku_mode=16

    save_warped_image(warped_image,processed_image)
    # cv2.imwrite(processed_image, warped_image)
    if sudoku_mode==16:
        save_warped_image(warped_image,image_to_model)
    return sudoku_mode
    # print('sudoku mode',sudoku_mode)


def save_warped_image(image, path):
    height, width,_ = image.shape
    square_size = max(height, width)
    square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    scale_x = square_size / width
    scale_y = square_size / height

    stretched_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)

    x_offset = (square_size - stretched_image.shape[1]) // 2
    y_offset = (square_size - stretched_image.shape[0]) // 2

    # stretched_image = cv2.cvtColor(stretched_image, cv2.COLOR_GRAY2BGR)

    square_image[
        y_offset: y_offset + stretched_image.shape[0],
        x_offset: x_offset + stretched_image.shape[1],
    ] = stretched_image
    resized_image = cv2.resize(square_image, (960, 960))

    cv2.imwrite(path, resized_image)
    # matrix = [[0 for _ in range(sudoku_mode)] for _ in range(sudoku_mode)]

    # read_sudoku_grid(warped_image,sudoku_mode,matrix)





