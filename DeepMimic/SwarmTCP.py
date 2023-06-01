import torch

from TCPenv import TCPenv
import random
from PTorchEnv.StateNormalize import WelFord_Normalizer
class SwarmTCP(TCPenv):
    def __init__(self,port,IP):
        super(SwarmTCP, self).__init__(port,IP)
        self.set_simer(self)
        self.steps=0
        self.normalizer=WelFord_Normalizer()
        self.limierror = 1

    def getreward(self,state,action):
        # reward=state[2,2]*0.5+state[2,8]*0.5+0.1*state[3,0]

        if(abs(state[1,0])<self.limierror and abs(state[1,1])<self.limierror
        and abs(state[1,3])<self.limierror and abs(state[1,4])<self.limierror and
        abs(state[1,6])<self.limierror and abs(state[1,7])<self.limierror):
            reward=1
        else:
            reward=0
        #模仿2D推箱子的格式,首先完成模型预训练

        return reward
    def calcobs(self,statevector):
        new_vec=statevector[1,:]
#第二列的所有元素才是输入，所以采用第二列
        return  new_vec
    def Info_extract(self,statevector):
        maximum_error=statevector[2,5]
        self.steps+=1
        if(self.steps>200):#或许不应该设置最大步数,奖励类比pendulum
            self.done=1
            self.steps=0
        else:
            self.done=0
        info="not_truncated"
        return info
    def missiondonejudge(self,statevector):

        return self.done
    def reset(self):
        state=torch.Tensor([[1]])
        self.Matadatamanager.sendMat(state)
        if(self.Matdatapointer.is_pudated()):##callback called
            callbackinfo=self.Matdatapointer.getdata()
            return self.calcobs(callbackinfo)
    def randonsample(self):
        pass #do nothing here