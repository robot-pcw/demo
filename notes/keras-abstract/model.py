#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/16/21 2:00 上午
# @Author  : 4c1p
# @Description: <>

import numpy as np
from operator import add
from functools import reduce
from .layers.layer import Layer, get_default_graph

class Model(Layer):
    """Model是特殊的'复合层'，支持训练/预测，也可继续与其他层相叠加
    """
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self._is_compiled = False
        self._layers = []
        self.optimizer = None

        self._losses = []
        self._metrics = []

    def compile(self, optimizer, loss=None, metrics=None):
        self.optimizer = optimizer # 参数更新策略
        self.loss = loss or {}  # 模型选择策略
        self.metrics = metrics or {}

    def show(self):
        for lay in self._layers:
            print(lay)

    def fit(self, x=None, y=None, validation_data=None, batch_size=None, epochs=1):
        context_graph = get_default_graph()
        layers_topo_order = context_graph.topo_sort()
        # 1) call layer in topo order

        # 2) update losses
        self._update_losses()
        loss = self._sum_losses()

        # 3) gradient optimize in reverse topo order


    def _update_losses(self):
        pass

    def evaluate(self, x=None, y=None, batch_size=None,):
        pass

    def _sum_losses(self):
        return reduce(add, self._losses)

    def predict(self, x, batch_size=None,):
        pass