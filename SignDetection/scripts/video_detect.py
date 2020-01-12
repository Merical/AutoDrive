import cv2
import numpy as np
import os
import time
from SignDetection.scripts.SignDetectionPack.models import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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


def split_frame_size(frame: np.ndarray, cell_rows: int, cell_cols: int, row_nums: int, col_nums: int):
    imgs = []
    rects = []

    img_rows, img_cols, _ = frame.shape # 1920 x 1080

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

def detect_region(frame:np.ndarray, center_row:int, center_col:int, row_nums:int, col_nums:int):
    imgs = []
    rects = []

    imgs.append(frame[center_row-row_nums//2:center_row+row_nums//2, center_col-col_nums//2:center_col+col_nums//2])
    rects.append(((center_col-col_nums//2, center_row-row_nums//2), (center_col+col_nums//2, center_row+row_nums//2)))

    center_col -= col_nums
    imgs.append(frame[center_row-row_nums//2:center_row+row_nums//2, center_col-col_nums//2:center_col+col_nums//2])
    rects.append(((center_col-col_nums//2, center_row-row_nums//2), (center_col+col_nums//2, center_row+row_nums//2)))

    # center_col -= col_nums//2
    # imgs.append(frame[center_row-row_nums//2:center_row+row_nums//2, center_col-col_nums//2:center_col+col_nums//2])
    # rects.append(((center_col-col_nums//2, center_row-row_nums//2), (center_col+col_nums//2, center_row+row_nums//2)))

    return imgs, rects


count = 0
frame_cout = 0
videos = os.listdir(video_dir)
with_gpu = True

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# detector = SignDetector("best_weights_1208.pth", with_gpu=True)
detector = SignDetector("shufflenet_v2_0_5_best_weights.pth", with_gpu=True)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

for v in videos :
    print('Processing video: {}'.format(v))
    cap = cv2.VideoCapture(os.path.join(video_dir, v))
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1080), True)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 480))
    frame_rows, frame_cols, _ = frame.shape
    if not ret:
        print("Error, the video is damaged!")

    img_rows, img_cols, _ = frame.shape # 1920 x 1080
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    while ret:

        tic = time.time()
        # frame = cv2.resize(frame, (640, 480))
        # imgs, rects = split_frame_size(frame.copy(), 448, 224, 1, 1)
        imgs, rects = detect_region(frame.copy(), frame_rows//2-320, frame_cols//2, 320, 240)
        # imgs, rects = detect_region(frame.copy(), frame_rows//2-120, frame_cols//2, 112, 56)
        results, preds = detector.detect(imgs)
        toc = time.time()
        print("Result is {0}, with time consumption {1:.2f} seconds.".format(results, toc-tic))
        for i in range(len(preds)):
            if preds[i]:
                cv2.rectangle(frame, rects[i][0], rects[i][1], (255, 0, 0), thickness=2)
                cv2.putText(frame, results[i], (rects[i][0][0] + 20, rects[i][0][1] - 20), 1, 2, (255, 0, 255), 2)
            # else:
            #     cv2.rectangle(frame, rects[i][0], rects[i][1], (0, 0, 255), thickness=2)

        cv2.imshow('Frame', frame)
        out.write(frame)
        key = cv2.waitKey(20)
        if key == ord('q'):
            break

        ret, frame = cap.read()

    break
cap.release()
out.release()
cv2.destroyAllWindows()