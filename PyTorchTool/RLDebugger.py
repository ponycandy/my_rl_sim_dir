import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from math import *
from PTorchEnv.Typechecker import double_typecheck
class RLDebugger():
    def __init__(self):
        plt.ion() #开启interactive mode 成功的关键函数
        plt.figure(1)
        self.counter=0
    def add_a_point(self,x,y,delay=0.05,color_0 = 'r'):
        y=double_typecheck(y)
        plt.plot(x,y,'.',color=color_0)
        plt.pause(delay)
        self.counter+=1
        # if (self.counter%50==0):
        #     plt.cla()
            # plt.axis([x, x+50, -10, 10])
