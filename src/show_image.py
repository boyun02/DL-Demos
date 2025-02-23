#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2025/2/23 9:36
# @Author :wyb
import cv2


def show_image_opencv(image_path):
    # 读取图片 (BGR格式)
    img = cv2.imread(image_path)

    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return

    # 转换为RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建可调整窗口
    cv2.namedWindow('OpenCV Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('OpenCV Image', 800, 600)  # 设置初始窗口尺寸

    # 显示图片
    cv2.imshow('OpenCV Image', img_rgb)
    cv2.waitKey(0)  # 等待按键
    cv2.destroyAllWindows()  # 关闭所有窗口


# 使用示例
show_image_opencv('../data/archive/dataset/training_set/cats/cat.1.jpg')