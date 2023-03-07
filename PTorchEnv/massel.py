import numpy as np

from Pyenv import Pyenv
import collections
class massel(Pyenv):
    def __init__(self):
        super(massel, self).__init__()
        self.set_simer(self)
        self.statenow=0
    def getreward(self,state,action):
        if action==1:
            return 1
        else:
            return -1
    def calcobs(self,statevector):
        #这一次是保持0位，所以无需处置,但是，不能够一致输出
        #这个函数总是需要考虑到statevector最后一位是标志位的情况
        return  statevector[0,:]
    def step_in(self,actionin):
        self.statenow[0,0]+=actionin[0,0]
        print("current_posision : ",self.statenow[0,0])
        return self.statenow
    def missiondonejudge(self,statenext):
        if statenext[0,0]==5:
            return 1
        else:
            return 0
    def sampleaction(self):
        return np.mat(collections.random())
    # def setstate_(self,state):
    #     self.statenow=state[0,0]
    #     return 1
# getreward(statenext,actionin)
# Info_extract(self.statenext)
# missiondonejudge(self.statenext)