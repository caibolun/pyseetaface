#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-05-06 17:20:35
@LastEditTime: 2020-07-21 22:35:13
'''
import libseetaface
import os
from os import path as osp
get_module_res = lambda *res: os.path.normpath(os.path.join(
                        os.getcwd(), os.path.dirname(__file__), *res))
_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))



class SeetaFace(object):
    def __init__(self, model_path=None, device="cpu", gpu_id=0):
        if model_path is None:
            self.model_path = get_module_res("model")
        else:
            self.model_path = _get_abs_path(model_path)
        self.fd_path = osp.join(self.model_path, 'fd_2_00.dat')
        self.fl81_path = osp.join(self.model_path, 'pd_2_00_pts81.dat')
        self.fl5_path = osp.join(self.model_path, 'pd_2_00_pts5.dat')
        self.fr_path = osp.join(self.model_path, 'fr_2_10.dat')
        self.device = device
        self.gpu_id = gpu_id

        self.detect_init = False
        self.align81_init = False
        self.align5_init = False
        self.extract_init = False
        self.pipeline = libseetaface.SeetaFaceAPI()
    
    def init(self):
        self.pipeline.init(self.fd_path, self.fl81_path, self.fl5_path, self.fr_path, self.device, self.gpu_id)


    def detect(self, img):
        if self.detect_init is False:
            self.pipeline.detect_init(self.fd_path, self.device, self.gpu_id)
            self.detect_init = True
        return self.pipeline.detect(img)

    def align81(self, img, rects):
        if self.align81_init is False:
            self.pipeline.align81_init(self.fl81_path, self.device, self.gpu_id)
            self.align81_init = True
        return self.pipeline.align81(img, rects)

    def align5(self, img, rects):
        if self.align5_init is False:
            self.pipeline.align5_init(self.fl5_path, self.device, self.gpu_id)
            self.align5_init = True
        return self.pipeline.align5(img, rects)
    
    def extract(self, img, rect=None, mark=None):
        if self.extract_init is False:
            self.pipeline.extract_init(self.fr_path, self.device, self.gpu_id)
            self.extract_init = True
        if mark is not None:
            return self.pipeline.extract(img, mark)
        if rect is not None:
            marks = self.pipeline.align5(img, [rect])
            mark = marks[0]
            return self.pipeline.extract(img, mark)

    def evaluate(self, img, rect, mark=None, face_size=80):
        if mark is None or len(mark) != 5:
            marks = self.pipeline.align5(img, [rect])
            mark = marks[0]
        return self.pipeline.evaluate(img, rect, mark, face_size)

    
    
    