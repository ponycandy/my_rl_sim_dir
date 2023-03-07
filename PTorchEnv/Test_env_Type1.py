import random

import numpy as np

from Pyenv import Pyenv
import collections
class Test_env_Type1(Pyenv):
    def __init__(self):
        super(Test_env_Type1, self).__init__()
        self.set_simer(self)
        self.statenow=1
        self.state_cache=0
        self.stepdone=0
    def getreward(self,state,action):
        if self.state_cache==1:
            self.state_cache=state[0,0]
            return 1
        else:
            self.state_cache=state[0,0]
            return -1
    def calcobs(self,statevector):
        #这一次是保持0位，所以无需处置,但是，不能够一致输出
        #这个函数总是需要考虑到statevector最后一位是标志位的情况
        return  statevector[0,0]
    def step_in(self,actionin):
        num=random.randint(0,1)
        if num==0:
            self.statenow=-1
            # print("rand obs : ",self.statenow)
            return self.statenow
        else:
            self.statenow= 1
            # print("rand obs : ",self.statenow)
            return self.statenow

    def missiondonejudge(self,statenext):
        return 1
    def sampleaction(self):
        return np.mat(collections.random())
    # def setstate_(self,state):
    #     self.statenow=state[0,0]
    #     return 1
# getreward(statenext,actionin)
# Info_extract(self.statenext)
# missiondonejudge(self.statenext)