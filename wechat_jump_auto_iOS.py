# -*- coding: utf-8 -*-

# 修改自 https://github.com/wangshub/wechat_jump_game/blob/master/wechat_jump_auto_iOS.py

"""
# === 思路 ===
# 核心：每次落稳之后截图，根据截图算出棋子的坐标和下一个块顶面的中点坐标，
#      根据两个点的距离乘以一个时间系数获得长按的时间
# 识别棋子：OpenCV 精准匹配
# 识别白点：OpenCV 精准匹配
# 识别棋盘：靠底色和方块的色差来做，从分数之下的位置开始，一行一行扫描，
           由于圆形的块最顶上是一条线，方形的上面大概是一个点，所以就
           用类似识别棋子的做法多识别了几个点求中点，这时候得到了块中点的 X
           轴坐标，这时候假设现在棋子在当前块的中心，根据一个通过截图获取的
           固定的角度来推出中点的 Y 坐标
# 最后：根据两点的坐标算距离乘以系数来获取长按时间（似乎可以直接用 X 轴距离）
"""
import os
import shutil
import time
import random
import json
from PIL import Image, ImageDraw
import wda
import math
import numpy as np
import cv2

with open('config.json', 'r') as f:
    config = json.load(f)

dev = config['dev']

# 长按的时间系数，请自己根据实际情况调节
press_coefficient = config['press_coefficient']
time_coefficient = config['press_coefficient']
# 棋子高度，要调节
piece_body_height = config['piece_body_height']
# 棋子的宽度，比截图中量到的稍微大一点比较安全，要调节
piece_body_width = config['piece_body_width']
#棋子同步到底座中部的距离，要调节
piece_top_to_base_height = config[ "piece_top_to_base_height"]


c = wda.Client()
s = c.session()



def pull_screenshot():
    c.screenshot('1.png')


def jump(distance):
    press_time = distance * time_coefficient / 1000
    print('press time: {}'.format(press_time))
    s.tap_hold(random.uniform(0, 320), random.uniform(64, 320), press_time)




temp1 = cv2.imread('temp_player.png', 0)
w1, h1 = temp1.shape[::-1]
# 匹配游戏结束画面的模板
temp_end = cv2.imread('temp_end.png', 0)
# 匹配中心小圆点的模板
temp_white_circle = cv2.imread('temp_white_circle.png', 0)
w2, h2 = temp_white_circle.shape[::-1]

def get_center(img_canny, ):
    H, W = img_canny.shape
    # 利用边缘检测的结果寻找物块的上沿和下沿
    # 进而计算物块的中心点
    y_top = np.nonzero([max(row) for row in img_canny[400:]])[0][0] + 400
    x_top = int(np.mean(np.nonzero(img_canny[y_top])))

    y_bottom = y_top + 50
    for row in range(y_bottom, H):
        if img_canny[row, x_top] != 0:
            y_bottom = row
            break

    x_center, y_center = x_top, (y_top + y_bottom) // 2
    return img_canny, x_center, y_center


def find_piece_and_board(im):

    _im = im

    # 如果在游戏截图中匹配到带"再玩一局"字样的模板，则循环中止
    res_end = cv2.matchTemplate(im, temp_end, cv2.TM_CCOEFF_NORMED)
    print(cv2.minMaxLoc(res_end)[1])
    if cv2.minMaxLoc(res_end)[1] > 0.95:
        print('Game over!')
        cv2.imwrite('Resault.png', _im)
        return 0,0,0,0

    # 模板匹配截图中小跳棋的位置

    res1 = cv2.matchTemplate(im, temp1, cv2.TM_CCOEFF_NORMED)
    min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
    piece_x, piece_y = (max_loc1[0]+27, max_loc1[1]+piece_top_to_base_height)

    # 先尝试匹配截图中的中心原点，
    # 如果匹配值没有达到0.95，则使用边缘检测匹配物块上沿
    res2 = cv2.matchTemplate(im, temp_white_circle, cv2.TM_CCOEFF_NORMED)
    min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
    if max_val2 > 0.95:
        print('found white circle!')
        board_x, board_y = max_loc2[0] + w2 // 2, max_loc2[1] + h2 // 2
    else:
        # 边缘检测
        im = cv2.GaussianBlur(im, (5, 5), 0)
        img_canny = cv2.Canny(im, 1, 10)
        H, W = img_canny.shape

        # 消去小跳棋轮廓对边缘检测结果的干扰
        for k in range(max_loc1[1] - 10, max_loc1[1] + piece_body_height):
            for b in range(max_loc1[0] - 10, max_loc1[0] + piece_body_width):
                img_canny[k][b] = 0

        im, board_x, board_y = get_center(img_canny)

    board_y = int(piece_y - abs(board_x - piece_x) * math.sqrt(3) / 3)
    
    if dev:
        # 将图片输出以供调试
        im = cv2.circle(_im, (board_x, board_y), 10, 255, -1)
        im = cv2.circle(_im, (piece_x, piece_y), 10, 255, -1)
        # cv2.rectangle(img_canny, max_loc1, (board_x, board_y), 255, 2)
        cv2.imwrite('./dev/ %s .png' % str((piece_x, piece_y, board_x, board_y)), _im)

    
    return piece_x, piece_y, board_x, board_y


def main():
    while True:
        pull_screenshot()
        im = cv2.imread('1.png',0)
        # im = Image.open("./1.png")

        # 获取棋子和 board 的位置
        piece_x, piece_y, board_x, board_y = find_piece_and_board(im)
        ts = int(time.time())
        print(ts, piece_x, piece_y, board_x, board_y)
        if piece_x == 0:
            return

        distance = (
            (board_x - piece_x) ** 2 + (board_y - piece_y) ** 2)**0.5

        jump(distance)

        # 为了保证截图的时候应落稳了，多延迟一会儿，随机值防 ban
        time.sleep(random.uniform(1, 1.1))


if __name__ == '__main__':
    main()
