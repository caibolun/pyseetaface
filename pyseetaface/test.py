#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-05-06 21:55:50
@LastEditors: xiaoyijia
@LastEditTime: 2020-06-29 17:31:27
'''
import seetaface
import cv2
model = seetaface.SeetaFace()
img = cv2.imread("./face.jpg", cv2.IMREAD_COLOR)
rects, scores = model.detect(img)
mark81 = model.align81(img, rects)
mark5 = model.align5(img, rects)

for rect, score, points81, points5 in zip(rects, scores, mark81, mark5):

    # evaluate quality
    res = model.evaluate(img, rect, points5)
    ret = res['ret']
    clarity = res['clarity']
    lightness = res['lightness']
    face_size = res['face_size']
    roll = res['roll']
    yaw = res['yaw']
    pitch = res['pitch']

    feat = model.extract(img, mark=points5)

    img = cv2.rectangle(img, rect, (0, 0, 255), 2)
    img = cv2.putText(img, "%.2f"%score, (rect[0], rect[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
    img = cv2.putText(img, "%.2f"%clarity, (rect[0], rect[1]+rect[3]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    for p in points81:
        img = cv2.circle(img, (int(p[0]), int(p[1])), 1, (255, 0, 0), 1)
    for p in points5:
        img = cv2.circle(img, (int(p[0]), int(p[1])), 1, (0, 255, 0), 1)

cv2.imwrite("result.jpg", img)
