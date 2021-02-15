#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/16/21 3:27 上午
# @Author  : 4c1p
# @Description: <>

class Loss(object):
    """Loss base class.
    """
    def __init__(self, reduction, name=None):
        self.reduction = reduction
        self.name = name

    def __call__(self, y_true, y_pred):
        loss = self.call(y_true, y_pred)
        return self._losses_reduce(loss)

    def _losses_reduce(self, loss):
        pass

    def call(self, y_true, y_pred):
        raise NotImplementedError