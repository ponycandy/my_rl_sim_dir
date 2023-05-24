import numpy as np

from Pyenv import Pyenv
from  DroneVrep import DroneVrep
import collections
class DroneEnv(Pyenv):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.set_simer(self)
        self.drone=DroneVrep()
        #总是使用statenow表示当前状态，请注意：
    # statenow同样被parent级调用，所以不要更改变量名称！！
    def getreward(self,state,action):
        if action[0,0]==1:
            return 1
        else:
            return -1
    def calcobs(self,statevector):
        #这一次是保持0位，所以无需处置,但是，不能够一致输出
        #这个函数总是需要考虑到statevector最后一位是标志位的情况
        return  statevector[0,:]
    def step_in(self,actionin):
        # 行动为1*4矩阵

        vel_list=[actionin[0,0],actionin[0,1],actionin[0,2],actionin[0,3]]
        self.drone.set_speed_list(vel_list)
        linear,angular=self.drone.get_drone_vel()
        pos=self.drone.get_drone_pos()
        orien=self.drone.get_drone_orien()
        self.statenow=[linear,angular,pos,orien]
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