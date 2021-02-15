#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/15/21 4:58 下午
# @Author  : 4c1p
# @Description: <layer interface>

import numpy as np
from tensorflow.keras import Model

"""
https://keras.io/api/layers/base_layer/
"""

DEFAULT_DTYPE = np.float32

class Layer:
    """This is the class from which all layers inherit.
    """

    def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
        self.name = name
        self.built = False # whether build needs to be called upon layer call
        self.dtype = DEFAULT_DTYPE if dtype is None else dtype

        self._trainable = trainable
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._updates = [] # 当前层的update ops list
        self._stateful = False

        batch_size, input_shape = kwargs.get('batch_size', None), kwargs.get('input_shape', None)
        self._batch_input_shape = None
        if batch_size and input_shape:
            self._batch_input_shape = (batch_size,) + tuple(input_shape)
        # self._initial_weights = kwargs.get('weights', None)
        self._inbound_nodes = []   # 输入层节点列表
        self._outbound_nodes = []  # 输出层节点列表
        self._losses = []  # 使用正则约束时产生的loss

        self._add_to_graph()

    def __call__(self, inputs, *args, **kwargs):
        """Wraps `call`: 若层变量未初始化，则先调用build()
        """
        if not self.built:
            input_shape = map(lambda x: x.shape, inputs) # 默认inputs中元素有shape属性
            self.build(input_shape)
        return self.call(inputs, *args, **kwargs)

    def build(self, input_shape):
        """层变量创建 (参数层需要重写)
        """
        self.built = True

    def call(self, inputs, *args, **kwargs):
        """layer compute: inputs -> outputs
        """
        raise NotImplementedError

    def _add_to_graph(self):
        """自动将当前层加入上下文计算图中
        """
        graph = get_default_graph()
        graph.add_node(self)

    def add_weights(self):
        pass


class Node(object):
    """A node from layer A to layer B is added to:
    - A._outbound_nodes
    - B._inbound_nodes
    """


class Graph(object):
    """计算图管理器，以layer对象为节点，layer._inbound_nodes为边关系
    """
    def __init__(self, name="default_graph", ):
        self.name = name
        self.graph_nodes = []

    def add_node(self, layer):
        self.graph_nodes.append(layer)

    def topo_sort(self):
        # layer节点之间的依赖关系进行拓扑排序，之后将按拓扑顺序进行前向计算
        def dfs(node, visited, topo_order):
            if node in visited:
                return
            visited.add(node)
            for n in node._inbound_nodes:
                dfs(n, visited, topo_order)
            topo_order.append(node)

        visited = set()
        topo_order = []
        for node in self.graph_nodes:
            dfs(node, visited, topo_order)
        return topo_order


def get_default_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = Graph()
    return _GRAPH