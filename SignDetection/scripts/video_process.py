import cv2
import numpy as np
import os

video_dir = "../videos"
output_dir = "../data/images"

def split_frame_num(frame: np.ndarray, row_nums: int, col_nums: int):
    imgs = []
    rects = []

    img_rows, img_cols, _ = frame.shape # 1920 x 1080
    cell_rows = img_rows // row_nums
    cell_cols = img_cols // col_nums

    for r in range(row_nums):
        for c in range(col_nums):
            rects.append(((cell_cols * c, cell_rows * r), (cell_cols * c + cell_cols, cell_rows * r + cell_rows)))
            imgs.append(frame[cell_rows * r : cell_rows * r + cell_rows, cell_cols * c : cell_cols * c + cell_cols])

    return imgs, rects


def split_frame_size(frame: np.ndarray, cell_rows: int, cell_cols: int):
    imgs = []
    rects = []

    img_rows, img_cols, _ = frame.shape # 1920 x 1080
    row_nums = img_rows // cell_rows
    col_nums = img_cols // cell_cols

    bias_row = (img_rows - row_nums * cell_rows) // 2
    bias_col = (img_cols - col_nums * cell_cols) // 2

    for r in range(row_nums):
        for c in range(col_nums):
            rects.append(((cell_cols * c + bias_col, cell_rows * r + bias_row),
                          (cell_cols * c + cell_cols + bias_col, cell_rows * r + cell_rows + bias_row)))
            imgs.append(frame[cell_rows * r + bias_row: cell_rows * r + cell_rows + bias_row,
                        cell_cols * c + bias_col: cell_cols * c + cell_cols + bias_col])
            if c != col_nums - 1 and r != row_nums - 1:
                rects.append(((cell_cols * c + bias_col + cell_cols // 2, cell_rows * r + bias_row + cell_rows //2),
                              (cell_cols * c + cell_cols + bias_col + cell_cols // 2, cell_rows * r + cell_rows + bias_row + cell_rows // 2)))
                imgs.append(frame[cell_rows * r + bias_row + cell_rows // 2: cell_rows * r + cell_rows + bias_row + cell_rows // 2,
                            cell_cols * c + bias_col + cell_cols // 2: cell_cols * c + cell_cols + bias_col + cell_cols // 2])

    return imgs, rects

count = 0
frame_cout = 0
videos = os.listdir(video_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for v in videos :
    print('Processing video: {}'.format(v))
    cap = cv2.VideoCapture(os.path.join(video_dir, v))
    ret, frame = cap.read()
    if not ret:
        print("Error, the video is damaged!")

    img_rows, img_cols, _ = frame.shape # 1920 x 1080
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    while ret:

        imgs, rects = split_frame_size(frame.copy(), 320, 240)
        for rect in rects:
            cv2.rectangle(frame, rect[0], rect[1], (255, 0, 0), thickness=2)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(20)
        if key == ord('q'):
            exit(0)
        elif key == ord('s'):
            for im in imgs:
                cv2.imwrite(os.path.join(output_dir, "image_{0:08d}.png".format(count)), im)
                count += 1

        ret, frame = cap.read()
