import numpy as np
from numba import *
from PIL import *
import requests
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import math
import time
from numba import jit, cuda
import argparse
import json
import sys
from itertools import permutations
import pandas as pd
from PIL import Image
import os

from skimage.filters import threshold_local


import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

config = load_config(args.config)

img_path = config['image_path']
block_size = tuple(config['block_size'])
min_thres = config['min_thres']
max_thres = config['max_thres']
brightness = config['brightness']
contrast = config['contrast']
gau_kernel_size = 3

img = [np.asarray(Image.open(os.path.join(img_path, x))) for x in os.listdir(img_path)]
gau_kernel = cv2.getGaussianKernel(gau_kernel_size**2,1)
gau_kernel = gau_kernel.reshape((gau_kernel_size,gau_kernel_size))
sobel_x_kernel, sobel_y_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32), np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)

def find_contours(canny_egde):
    contours, _ = cv2.findContours(canny_egde, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)
    chosen_contour = contours[0].squeeze()

    max_x, max_y = chosen_contour[:, 0].max(), chosen_contour[:, 1].max()
    min_x, min_y = chosen_contour[:, 0].min(), chosen_contour[:, 1].min()

    bottom_contour, top_contour = chosen_contour[chosen_contour[:, 0] == max_x], chosen_contour[chosen_contour[:, 0] == min_x]
    left_contour, right_contour = chosen_contour[chosen_contour[:, 1] == min_y], chosen_contour[chosen_contour[:, 1] == max_y]

    top, bottom = get_pos(top_contour, min_y, max_y, 1), get_pos(bottom_contour, min_y, max_y, 1)
    left, right = get_pos(left_contour, min_x, max_x, 0), get_pos(right_contour, min_x, max_x, 0)
    return top, bottom, left, right
    
    
def get_pos(lst_can, min_val, max_val, pos=0):
    """
    Xác định các vị trí trong danh sách thỏa mãn giá trị tối thiểu và tối đa cho một trục cụ thể.

    Tham số:
    - lst_can: Danh sách các vị trí cần kiểm tra.
    - min_val: Giá trị tối thiểu.
    - max_val: Giá trị tối đa.
    - pos: Vị trí trục cần kiểm tra.

    """
    candidates = lst_can[(lst_can[:, pos] == min_val) | (lst_can[:, pos] == max_val)]

    if len(candidates) == 0:
        return lst_can[lst_can[:, pos].argmax()], -1
    elif len(candidates) == 2:
        return candidates[candidates[:, pos] == min_val][0], candidates[candidates[:, pos] == max_val][0]
    elif len(candidates) == 1:
        return candidates[candidates[:, pos] == max_val][0], -1
    else:
        return candidates[candidates[:, pos] == min_val][0], -1


def get_points(xy, lst):
    """
    Thêm các điểm vào danh sách.

    Tham số:
    - xy: Các tọa độ cần thêm vào danh sách.
    - lst: Danh sách cần thêm vào.

    """
    if xy[1] == -1:
        lst.append(xy[0])
    else:
        lst.extend(xy)

def get_4_vertices(t, l, r, b, lst_v):
    """
    Xác định 4 điểm đỉnh từ các cạnh của hình.

    Tham số:
    - t: Cạnh trên.
    - l: Cạnh trái.
    - r: Cạnh phải.
    - b: Cạnh dưới.
    - lst_v: Danh sách chứa các điểm đỉnh.

    """
    lst = []

    get_points(list(t), lst)
    get_points(list(l), lst)
    get_points(list(r), lst)
    get_points(list(b), lst)

    lst_v.extend([point for point in lst if point not in lst_v])


@jit(cache = True)
def get_l_r(lst, pts, is_top=True):
    """
    Xác định hai điểm trái và phải từ một danh sách và một tập hợp các điểm.

    Tham số:
    - lst: Danh sách cần kiểm tra.
    - pts: Tập hợp chứa các điểm.
    - is_top: Biến bool xác định xem các điểm là trên cùng hay không.

    Trả về:
    Hai điểm trái và phải.
    """
    l = []
    for i in lst:
        for v in pts:
            if i == v[0]:
                l.append(v)
    l.sort(key=lambda x: x[1])
    if is_top:
        return l[0], l[1]
    else:
        return l[1], l[0]

@jit(cache = True)
def order_points_py(pts):
    """
    Sắp xếp các điểm theo thứ tự cụ thể.

    Tham số:
    - pts: Danh sách các điểm cần sắp xếp.

    Trả về:
    Danh sách các điểm đã được sắp xếp.
    """
    t, b = np.sort(pts[:, 0])[:2], np.sort(pts[:, 0])[2:]

    order_vertices = []
    order_vertices.extend(list(get_l_r(t, pts)))
    order_vertices.extend(list(get_l_r(b, pts, False)))

    return np.array(order_vertices)

def order_points_np(pts):
    """
    Sắp xếp các điểm theo thứ tự cụ thể.

    Tham số:
    - pts: Danh sách các điểm cần sắp xếp.

    Trả về:
    Danh sách các điểm đã được sắp xếp.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Tính tổng các tọa độ của mỗi điểm
    s = pts.sum(axis=1)
    # Lấy điểm có tổng nhỏ nhất và lớn nhất
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Tính hiệu giữa hai tọa độ của mỗi điểm
    diff = np.diff(pts, axis=1)
    # Lấy điểm có hiệu nhỏ nhất và lớn nhất
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
    
def calculate_dimensions(vertices):
    top_left, top_right, bottom_right, bottom_left = vertices
    widthA = ((bottom_right[0] - bottom_left[0]) ** 2 + (bottom_right[1] - bottom_left[1]) ** 2) ** 0.5
    widthB = ((top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_left[1]) ** 2) ** 0.5
    maxWidth_py = int(widthA) if int(widthA) > int(widthB) else int(widthB)

    heightA = ((top_right[0] - bottom_right[0]) ** 2 + (top_right[1] - bottom_right[1]) ** 2) ** 0.5
    heightB = ((top_left[0] - bottom_left[0]) ** 2 + (top_left[1] - bottom_left[1]) ** 2) ** 0.5

    maxHeight_py = int(heightA) if int(heightA) > int(heightB) else int(heightB)

    return maxWidth_py, maxHeight_py
    
    
gray_cv = [cv2.cvtColor(cv2.cvtColor(x,cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY) for x in img]
gaussianBlur_cv = [cv2.GaussianBlur(gray_cv[i],(gau_kernel_size, gau_kernel_size),1, borderType = cv2.BORDER_REPLICATE)  for i in range(len(img))]
canny_edge_cv = [cv2.Canny(gaussianBlur_cv[i], 100, 200) for i in range(len(img))]    
results = [find_contours(canny_edge) for canny_edge in canny_edge_cv]
top, bottom, left, right = zip(*results)
four_vertices_cnts_cv = []
for i in range(len(top)):
    temp = []
    get_4_vertices(top[i], left[i], right[i], bottom[i], temp)
    four_vertices_cnts_cv.append(temp)
    
four_vertices_cnts_cv = [order_points_np(np.array(x)) for x in four_vertices_cnts_cv]
results = [calculate_dimensions(vertices) for vertices in four_vertices_cnts_cv]
maxWidth_cv, maxHeight_cv = zip(*results)

@cuda.jit
def convert_RGB_To_Gray_cuda(img_in, img_out):
    cols, rows = cuda.grid(2)
    if rows < img_in.shape[0] and cols < img_in.shape[1]:
        img_out[rows,cols] =  round(img_in[rows,cols,0] * 0.2989  + img_in[rows,cols,1] * 0.5870 + img_in[rows,cols,2] * 0.1141)

@cuda.jit
def convolution_cuda(img_in, img_out, kernel):
    c, r = cuda.grid(2)

    # Kiểm tra xem chỉ số hàng và cột của thread có nằm trong kích thước ảnh không
    if r < img_in.shape[0] and c < img_in.shape[1]:
        result = 0

        # Duyệt qua các phần tử trong kernel
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                # Tính toán chỉ số hàng và cột trong ảnh gốc dựa trên kernel và vị trí hiện tại của thread
                in_r = r - kernel.shape[0] // 2 + i
                in_c = c - kernel.shape[1] // 2 + j

                # Giới hạn chỉ số hàng và cột trong phạm vi của ảnh gốc
                in_r = min(max(0, in_r), img_in.shape[0] - 1)
                in_c = min(max(0, in_c), img_in.shape[1] - 1)

                # Tính tổng convolution
                result += kernel[i, j] * img_in[in_r, in_c]

        # Gán giá trị convolution đã tính được vào ảnh output, làm tròn kết quả
        img_out[r, c] = round(result)

@cuda.jit
def gradient_cuda(img_in, img_out, angle, kernel_x, kernel_y):
    """
    Tính toán gradient và hướng gradient của mỗi pixel trong ảnh đầu vào.

    Tham số:
    - img_in: Ảnh đầu vào.
    - img_out: Ma trận chứa gradient của mỗi pixel.
    - angle: Ma trận chứa hướng gradient của mỗi pixel.
    - kernel_x: Ma trận kernel cho gradient theo phương x.
    - kernel_y: Ma trận kernel cho gradient theo phương y.

    """
    c, r = cuda.grid(2)
    if r < img_in.shape[0] and c < img_in.shape[1]:
        # Khởi tạo giá trị gradient và hướng gradient
        result_x = 0
        result_y = 0
        # Tính toán gradient bằng cách áp dụng kernel
        for i in range(kernel_x.shape[0]):
            for j in range(kernel_x.shape[1]):
                in_r = r - kernel_x.shape[0] // 2 + i
                in_c = c - kernel_x.shape[1] // 2 + j
                in_r = min(max(0, in_r), img_in.shape[0] - 1)
                in_c = min(max(0, in_c), img_in.shape[1] - 1)
                result_x += kernel_x[i, j] * img_in[in_r, in_c]
                result_y += kernel_y[i, j] * img_in[in_r, in_c]
        # Tính toán magnitude và hướng của gradient
        img_out[r, c] = ((result_x ** 2) + (result_y ** 2)) ** 0.5
        angle[r, c] = math.atan2(result_y, result_x) * (180 / np.pi)
        angle[r, c] = angle[r, c] if angle[r, c] >= 0 else angle[r, c] + 180

@cuda.jit
def nonMaxSuppression_cuda(img_in, img_out, angle):
    c,r = cuda.grid(2)
    if 0 < r < img_in.shape[0] - 1 and 0 < c < img_in.shape[1]-1:
        q, t = 255, 255
        if 0 <= angle[r,c] < 22.5 or 180 >= angle[r,c] >= 157.5:
            q, t = img_in[r, c+1], img_in[r, c-1]
        elif 67.5 > angle[r,c] >= 22.5:
            q, t = img_in[r+1, c-1], img_in[r-1, c+1]
        elif 112.5 > angle[r,c] >= 67.5:
            q, t = img_in[r+1, c], img_in[r-1, c]
        elif 157.5 > angle[r,c] >= 112.5:
            q, t = img_in[r-1, c-1], img_in[r+1, c+1]

        if img_in[r,c] >=q and img_in[r,c] >=t:
            img_out[r,c] = img_in[r,c]
        else:
            img_out[r,c] = 0

@cuda.jit
def findMax(result, values):
    i,j = cuda.grid(2)
    if i < values.shape[1] and j < values.shape[0]:
        cuda.atomic.max(result, 0, values[j,i])

@cuda.jit
def thresholding_cuda(img_in, img_out, min_val, max_val):
    # Lấy chỉ số của thread
    c, r = cuda.grid(2)

    # Kiểm tra xem chỉ số của thread có nằm trong kích thước ảnh không
    if r < img_in.shape[0] and c < img_in.shape[1]:
        # Nếu giá trị pixel lớn hơn ngưỡng cao, đánh dấu là cạnh
        if img_in[r, c] > max_val:
            img_out[r, c] = 255
        # Nếu giá trị pixel nằm giữa ngưỡng thấp và ngưỡng cao
        elif img_in[r, c] > min_val:
            # Khởi tạo cờ đánh dấu cho việc xác định cạnh
            is_edge = False
            # Duyệt qua các pixel lân cận
            for i in range(max(0, r - 1), min(img_in.shape[0], r + 2)):
                for j in range(max(0, c - 1), min(img_in.shape[1], c + 2)):
                    # Nếu có pixel lân cận lớn hơn ngưỡng cao, đánh dấu là cạnh
                    if img_in[i, j] > max_val:
                        is_edge = True
                        img_out[r, c] = 255
                        break
                if is_edge:
                    break
        # Nếu giá trị pixel nhỏ hơn hoặc bằng ngưỡng thấp, không phải cạnh
        else:
            img_out[r, c] = 0

def transform_to_max_rectangle(img):
    cnts_jit, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_jit = sorted(cnts_jit, key = lambda x: cv2.arcLength(x,True), reverse = True)
    choosen_cnt = cnts_jit[0].squeeze()

    max_x, max_y = choosen_cnt[:,0].max(),choosen_cnt[:,1].max()
    min_x, min_y = choosen_cnt[:,0].min(), choosen_cnt[:,1].min()

    b_can, t_can = choosen_cnt[choosen_cnt[:,0] == max_x], choosen_cnt[choosen_cnt[:,0] == min_x]
    l_can, r_can = choosen_cnt[choosen_cnt[:,1] == min_y], choosen_cnt[choosen_cnt[:,1] == max_y]

    t, b = get_pos(t_can, min_y, max_y, 1), get_pos(b_can, min_y, max_y, 1)
    l, r = get_pos(l_can, min_x, max_x, 0), get_pos(r_can, min_x, max_x, 0)

    four_vertices_cnts_cuda = []

    get_4_vertices(t, l, r, b, four_vertices_cnts_cuda)

    four_vertices_cnts_cuda = order_points_py(np.array(four_vertices_cnts_cuda))#, max_x, max_y, min_x, min_y)

    tl, tr, br, bl = four_vertices_cnts_cuda
    widthA = ((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2) ** 0.5
    widthB = ((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2) ** 0.5

    maxWidth = int(widthA) if int(widthA) > int(widthB) else int(widthB)

    heightA = ((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2) ** 0.5
    heightB = ((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2) ** 0.5

    maxHeight = int(heightA) if int(heightA) > int(heightB) else int(heightB)
    n_four_vertices_cnts_cuda = np.array([[0,0], [0, maxWidth-1], [maxHeight-1, maxWidth - 1], [maxHeight - 1, 0]], dtype = 'float32')
    H_cuda = np.linalg.inv(cv2.getPerspectiveTransform(four_vertices_cnts_cuda.astype('float32'), n_four_vertices_cnts_cuda))
    return H_cuda, maxWidth, maxHeight, four_vertices_cnts_cuda.astype('int')

@cuda.jit
def map_homo_cuda(img_in, img_out, H):
    """
    Thực hiện ánh xạ các điểm từ ảnh đầu vào sang ảnh đầu ra bằng phép biến đổi Homography.

    Tham số:
    - img_in: Ảnh đầu vào.
    - img_out: Ảnh đầu ra sau khi ánh xạ.
    - H: Ma trận biến đổi Homography.

    """
    c, r = cuda.grid(2)
    if r < img_out.shape[0] and c < img_out.shape[1]:
        # Tính toán tọa độ mới của pixel trong ảnh đầu vào
        x = (H[0, 0] * c + H[0, 1] * r + H[0, 2]) / (H[2, 0] * c + H[2, 1] * r + H[2, 2])
        y = (H[1, 0] * c + H[1, 1] * r + H[1, 2]) / (H[2, 0] * c + H[2, 1] * r + H[2, 2])
        x = int(x) + 1 if (x * 100) % 100 >= 50 else int(x)
        y = int(y) + 1 if (y * 100) % 100 >= 50 else int(y)
        # Kiểm tra xem tọa độ mới có nằm trong phạm vi của ảnh đầu vào không
        if 0 <= y < img_in.shape[0] and 0 <= x < img_in.shape[1]:
            # Ánh xạ giá trị pixel từ ảnh đầu vào sang ảnh đầu ra
            img_out[r, c, 0] = img_in[y, x, 0]
            img_out[r, c, 1] = img_in[y, x, 1]
            img_out[r, c, 2] = img_in[y, x, 2]


def controller(img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    alpha = 1.0 + float(contrast) / 100
    beta = float(brightness)

    adjusted_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_image



start_time = time.time()
gray_cuda = [np.zeros((img[i].shape[0], img[i].shape[1])) for i in range(len(img))]

grid_size = [(math.ceil(gray_cuda[i].shape[1] / block_size[0]), #chieu x
             math.ceil(gray_cuda[i].shape[0] / block_size[1])) for i in range(len(img))] #chieu y
for i in range(len(img)):
    convert_RGB_To_Gray_cuda[grid_size[i], block_size](img[i].copy(), gray_cuda[i])

gaussian_cuda = [np.zeros(gray_cuda[i].shape) for i in range(len(img))]
for i in range(len(img)):
    convolution_cuda[grid_size[i], block_size](gray_cuda[i], gaussian_cuda[i], gau_kernel)

edge_grad_cuda = [np.zeros(gray_cuda[i].shape) for i in range(len(img))]
angle_cuda = [np.zeros(gray_cuda[i].shape) for i in range(len(img))]
for i in range(len(img)):
    gradient_cuda[grid_size[i], block_size](gaussian_cuda[i], edge_grad_cuda[i], angle_cuda[i], sobel_x_kernel, sobel_y_kernel)

non_max_cuda = [np.zeros(edge_grad_cuda[i].shape) for i in range(len(img))]
for i in range(len(img)):
    nonMaxSuppression_cuda[grid_size[i], block_size](edge_grad_cuda[i], non_max_cuda[i], angle_cuda[i])

max_val_cuda = [np.zeros(1, dtype=non_max_cuda[i].dtype) for i in range(len(img))]
for i in range(len(img)):
    findMax[grid_size[i],block_size](max_val_cuda[i], non_max_cuda[i])
    
max_val = [x * max_thres for x in max_val_cuda]
min_val = [x * min_thres for x in max_val]

canny_edge_cuda = [np.zeros(non_max_cuda[i].shape, np.int32) for i in range(len(img))]
canny_edge_cuda  = [canny_edge_cuda[i].astype(np.uint8) for i in range(len(img))]

max_val = np.concatenate(max_val)
min_val = np.concatenate(min_val)

for i in range(len(img)):
    thresholding_cuda[grid_size[i], block_size](non_max_cuda[i], canny_edge_cuda[i], min_val[i], max_val[i])

H_cuda, maxWidth, maxHeight, four_vertices_cnts_cuda = [], [], [], []
for i in range(len(img)):
    temp_H_cuda, temp_maxWidth, temp_maxHeight, temp_four_vertices_cnts_cuda = transform_to_max_rectangle(canny_edge_cuda[i])
    H_cuda.append(temp_H_cuda)
    maxWidth.append(temp_maxWidth)
    maxHeight.append(temp_maxHeight)
    four_vertices_cnts_cuda.append(temp_four_vertices_cnts_cuda)

output_4_points_cuda = [x.copy() for x in img]
_ = [cv2.drawContours(output_4_points_cuda[i], [four_vertices_cnts_cuda[i]], 0, (0, 255, 0), 2) for i in range(len(img))]

homo_img_cuda = []
for i in range(len(img)):
    if maxHeight[i] != maxWidth_cv[i] or maxWidth[i] != maxHeight_cv[i]:
        maxHeight[i] = maxWidth_cv[i]
        maxWidth[i] = maxHeight_cv[i]
        homo_img_cuda.append(np.zeros((maxWidth[i], maxHeight[i], img[i].shape[-1]), img[i].dtype))
    else:
        homo_img_cuda.append(np.zeros((maxWidth[i], maxHeight[i], img[i].shape[-1]), img[i].dtype))

# homo_img_cuda = [np.zeros((maxHeight_cv[i], maxHeight[i], img[i].shape[-1]), img[i].dtype) for i in range(len(img))]
grid_size = [(math.ceil(maxHeight[i] / block_size[0]), #chieu x
             math.ceil(maxWidth[i] / block_size[1])) for i in range(len(img))] #chieu y

for i in range(len(img)):
    map_homo_cuda[grid_size[i], block_size](img[i].copy(), homo_img_cuda[i], H_cuda[i])

homo_img_cuda = [homo_img_cuda[i].astype(int) for i in range(len(img))]
in_adjust = [homo_img_cuda[i].copy() for i in range(len(img))]
new_image_cuda = [controller(in_adjust[i], brightness, contrast) for i in range(len(img))]



end_time = time.time()
processed_time = end_time - start_time
print("\nProcessed time: ", processed_time,"s")
# image_out = img_path.split('.')[0] + '_result.jpg'
# plt.imsave(image_out, new_image_cuda.astype(np.uint8))