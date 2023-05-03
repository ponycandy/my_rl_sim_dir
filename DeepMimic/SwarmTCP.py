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

    def getreward(self,state,action):
        reward=state[2,2]*0.5+state[2,8]*0.5

        return reward
    def calcobs(self,statevector):
        new_vec=statevector[1,:]
#第二列的所有元素才是输入，所以采用第二列
        return  new_vec
    def Info_extract(self,statevector):
        maximum_error=statevector[2,5]
        self.steps+=1
        if(maximum_error<-300):#或许不应该设置最大步数,奖励类比pendulum
            self.done=1
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
            return callbackinfo