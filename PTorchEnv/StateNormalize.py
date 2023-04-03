import torch
import math
from PTorchEnv.Typechecker import TensorTypecheck
# 参照算法：https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
# 参照实现:https://github.com/siekmanj/r2l/blob/master/policies/base.py#L52
class WelFord_Normalizer():
    def __init__(self):
        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1
    def Reset(self):
        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1
    def normalize_state(self, state, update=True):
        """
        Use Welford's algorithm to normalize a state, and optionally update the statistics
        for normalizing states using the new state, online.
        """
        #我的输入必然是2D的列向量，所以这里做一些更改
        state = TensorTypecheck(state)
        #输出尺寸为1*n

        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1))
            self.welford_state_mean_diff = torch.ones(state.size(-1))
        state=state.squeeze(0)
        if update:
            if len(state.size()) == 1: # if we get a single state vector
                state_old = self.welford_state_mean
                self.welford_state_mean += (state - state_old) / self.welford_state_n
                self.welford_state_mean_diff += (state - state_old) * (state - state_old)
                self.welford_state_n += 1
            else:
                raise RuntimeError # this really should not happen
            get_state=(state - self.welford_state_mean) / torch.sqrt(self.welford_state_mean_diff / self.welford_state_n)
        return TensorTypecheck(get_state)