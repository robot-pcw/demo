#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/30/21 12:30 下午
# @Author  : 4c1p
# @Description: <>
from typing import List, NoReturn, Tuple
import numpy as np

class HMM:
    """
    states: 状态集合大小, 分别对应0~(n-1)这n个状态
    outputs: 即observations, 观测结果集合大小，分别对应0~(m-1)这M种状态
    A: 状态转移矩阵 N×N
    B: 观测概率矩阵 N×M
    """
    def __init__(self, states: int, outputs: int) -> NoReturn:
        self.states = states
        self.outputs = outputs

    def load(self, A: np.ndarray, B: np.ndarray, pi: np.ndarray) -> NoReturn:
        self.A = A
        self.B = B
        self.pi = pi

    def fit(self, outputs: np.ndarray) -> NoReturn:
        """todo: 使用EM算法从观测序列样本中估计模型参数
        """
        pass


    def observe_prob(self, outputs: List[int]) -> float:
        """已知观测序列，计算其出现概率
        """
        return self._forward(outputs)

    def _forward(self, outputs: List[int]) -> float:
        local_state_prob = None
        for ob in outputs:
            if local_state_prob is not None:
                local_state_prob = np.dot(self.A.T, local_state_prob)  #前置状态i转移到当前j的概率
                local_state_prob *= self.B[:, ob]
            else:
                local_state_prob = self.pi * self.B[:, ob]     #初始概率
        return np.sum(local_state_prob)


    def decode(self, outputs: List[int]) -> List[int]:
        """hidden state decode：找到最大可能的状态序列
        """
        return self._viterbi(outputs)

    def _viterbi(self, outputs: List[int]) -> Tuple[List[int], float]:
        """使用DP的维特比解码
        """
        T = len(outputs)
        state_path = [[-1]*self.states for _ in range(T)]   #record each step state prob
        prob = self.pi * self.B[:,outputs[0]]
        for t in range(1, T):
            prob_ = np.ones_like(prob)
            for i in range(self.states):
                prob_ji = prob * self.A[:,i].T    # state j -> state i
                prob_i, prefix_i = np.max(prob_ji) * self.B[i][outputs[t]], np.argmax(prob_ji)
                prob_[i] = prob_i
                state_path[t][i] = prefix_i
            prob = prob_
        decode_id = np.argmax(prob)
        decode = []
        step = T-1
        while decode_id >= 0:
            decode.append(decode_id)
            decode_id = state_path[step][decode_id]
            step -= 1
        return decode, max(prob)


def test_hmm():
    A = np.array([[0.5, 0.2, 0.3],[0.3, 0.5, 0.2],[0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    p0 = np.array([0.2, 0.4, 0.4])
    hmm = HMM(3, 2)
    hmm.load(A, B, p0)
    obs = [0, 1, 0]
    print("观测序列概率：", hmm.observe_prob(obs))
    print("最优状态序列及概率: ", hmm.decode(obs))


if __name__ == '__main__':
    test_hmm()




