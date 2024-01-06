import numpy as np
import cv2
import operator
import numpy as np
# import pytesseract
from matplotlib import pyplot as plt



def show_digits(digits,puzzle_size, colour=255):
    """Shows list of 81 extracted digits in a grid format"""
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(puzzle_size):
        row = np.concatenate(with_border[i * puzzle_size:((i + 1) * puzzle_size)], axis=1)
        rows.append(row)
    img = np.concatenate(rows)
    return img
 

# def convert_when_colour(colour, img):
#     """Dynamically converts an image to colour if the input colour is a tuple and the image is grayscale."""
#     if len(colour) == 3:
#         if len(img.shape) == 2:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#         elif img.shape[2] == 1:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     return img



def pre_process_image(img, skip_dilate=False):
    """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

    # Gaussian blur with a kernal size (height, width) of 9.
    # Note that kernal sizes must be positive and odd and the kernel must be square.
    proc = cv2.GaussianBlur(img.copy(), (5, 5), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert colours, so gridlines have non-zero pixel values.
    # Necessary to dilate the image, otherwise will look like erosion instead.
    proc = cv2.bitwise_not(proc, proc)

    if not skip_dilate:
        # Dilate the image to increase the size of the grid lines.
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        proc = cv2.dilate(proc, kernel)

    return proc


def find_corners_of_largest_polygon(img):
    """Finds the 4 extreme corners of the largest contour in the image."""
    opencv_version = cv2.__version__.split('.')[0]
    if opencv_version == '3':
        _, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    else:
        contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort by area, descending
    polygon = contours[0]  # Largest image

    # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
    # Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]



def crop_and_warp(img, crop_rect):
    """Crops and warps a rectangular section from an image into a square of similar size."""

    def distance_between(p1, p2):
        """Returns the scalar distance between two points"""
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a ** 2) + (b ** 2))

    # Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Get the longest side in the rectangle
    side = max([
    	distance_between(bottom_right, top_right),
    	distance_between(top_left, bottom_left),
    	distance_between(bottom_right, bottom_left),
    	distance_between(top_left, top_right)
    ])

    # Describe a square with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    return cv2.warpPerspective(img, m, (int(side), int(side)))


def infer_grid(img,puzzle_size):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[:1]
    side = side[0] / puzzle_size

    # Note that we swap j and i here so the rectangles are stored in the list reading left-right instead of top-down.
    for j in range(puzzle_size):
        for i in range(puzzle_size):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
    return squares


def cut_from_rect(img, rect):
    """Cuts a rectangle from an image using the top left and bottom right points."""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_centre(img, size, margin=0, background=0):
    """Scales and centres an image onto a new background square."""
    h, w = img.shape[:2]

    def centre_pad(length):
        """Handles centering for a given length that may be odd or even."""
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)
    
    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    """
    Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
    connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
    """
    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]   
    max_area = 0
    seed_point = (None, None)   
    if scan_tl is None:
        scan_tl = [0, 0]    
    if scan_br is None:
        scan_br = [width, height]   
    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
    	for y in range(scan_tl[1], scan_br[1]):
            # Only operate on light or white squares
            if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y) 
    # Colour everything grey (compensates for features outside of our middle scanning range
    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)    
    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image   
    # Highlight the main feature
    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)   
    top, bottom, left, right = height, 0, width, 0  
    for x in range(width):
    	for y in range(height):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0) 
    		# Find the bounding parameters
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right   
    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):
    """Extracts a digit (if one exists) from a Sudoku square."""

    digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

    # Use fill feature finding to get the largest feature in middle of the box
    # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)

    # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    # Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_centre(digit, size, 7)
    else:
        return np.zeros((size, size), np.uint8)


def get_digits(img, squares, size):
    """Extracts digits from their cells and builds an array"""
    digits = []
    img = pre_process_image(img.copy(), skip_dilate=True)
#    cv2.imshow('img', img)
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits


def process_image(path, puzzle_size):
    original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed = pre_process_image(original)
        
    corners = find_corners_of_largest_polygon(processed)
    cropped = crop_and_warp(original, corners)
        
    squares = infer_grid(cropped,puzzle_size)
    digits = get_digits(cropped, squares, 36)

    final_image = show_digits(digits,puzzle_size)
    return final_image


def extract_number(image_part):
    custom_config = r'--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
    custom_config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789 -l eng'
    custom_config = r'--psm 10 --oem 3 -l eng'
    # txt = pytesseract.image_to_string(image_part,config=custom_config)
    # return txt
    # cv2.imshow('',image_part)
    # cv2.waitKey(0)


def get_puzzle(image, puzzle_size):
    num_rows=puzzle_size
    num_cols=puzzle_size
    height, width = image.shape[:2]

    square_height = height // num_rows
    square_width = width // num_cols

    puzzle = []

    for i in range(num_rows):
        row = []  # Initialize a new row for each iteration
        for j in range(num_cols):
            y1 = i * square_height
            y2 = (i + 1) * square_height
            x1 = j * square_width
            x2 = (j + 1) * square_width
            square_part = image[y1:y2, x1:x2]
            s=extract_number(square_part)
            s='0' #if s == '' else s
            number = int(s)#.strip())
            row.append(0 if number is None else number)
        puzzle.append(row)

    return puzzle


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


def save_puzzle_to_file(puzzle,filename):
    with open(filename, 'w') as file:
        for row in puzzle:
            file.write(' '.join(map(str, row)) + '\n')


def save_warped_image(image,path):
    height, width = image.shape
    square_size = max(height, width)
    square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)

    scale_x = square_size / width
    scale_y = square_size / height

    stretched_image = cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y)

    x_offset = (square_size - stretched_image.shape[1]) // 2
    y_offset = (square_size - stretched_image.shape[0]) // 2


    stretched_image = cv2.cvtColor(stretched_image, cv2.COLOR_GRAY2BGR)

    square_image[
        y_offset : y_offset + stretched_image.shape[0],
        x_offset : x_offset + stretched_image.shape[1],
    ] = stretched_image
    resized_image = cv2.resize(square_image, (960, 960))

    cv2.imwrite(path, resized_image)


def run(original_image,save_image,extracted_puzzle):
    puzzle_size=9 
    puzzle_path=extracted_puzzle
    processed_image = process_image(original_image,puzzle_size)
    save_warped_image(processed_image,save_image)

    puzzle=get_puzzle(processed_image,puzzle_size)
    # for row in puzzle:
    #     print(row)
    # print(is_valid_sudoku(puzzle,puzzle_size))

    save_puzzle_to_file(puzzle, puzzle_path)


if __name__ == '__main__':
    run()

    # cv2.imwrite('recogniser\puzzle.jpg', image)

    # custom_config = r'--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
    # txt = pytesseract.image_to_string(image,config=custom_config)
    # print(txt)

    # Convert recognized text to a 2D array
    # puzzle = []
    # for i in range(puzzle_size):
    #     row = [int(c) if c.isdigit() else 0 for c in txt[i * puzzle_size:(i + 1) * puzzle_size]]
    #     puzzle.append(row)

    # # Print the 2D array
    # for row in puzzle:
    #     print(row)


    # show_image(image)
    # img=cv2.imread('recogniser/txt.jpg')
    # txt = pytesseract.image_to_string(img)
    # print(txt)
    # grid = extract_number(image)
    # print('Sudoku:')
    # display_sudoku(grid.tolist())
    # solution = sudoku_solver(grid)
    # print('Solution:')
    #    print(solution)  
    # display_sudoku(solution.tolist())