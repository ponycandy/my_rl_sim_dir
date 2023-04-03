from TCPenv import TCPenv
import random
from PTorchEnv.StateNormalize import WelFord_Normalizer
class CartpoleTCP(TCPenv):
    def __init__(self,port,IP):
        super(CartpoleTCP, self).__init__(port,IP)
        # self.set_pointee(self)
        self.set_simer(self)
        self.steps=0
        self.normalizer=WelFord_Normalizer()

    def getreward(self,state,action):
        phi=state[2,0]
        if(phi>3.1415926535):
            phi=2*3.1415926535-phi
        if(phi<-3.1415926535):
            phi=phi+2*3.1415926535
        if(abs(phi)>0.2):
            return 0
#debug终点状态必须返回0
        # if(abs(phi)<3.1415926535/10):
        #     return 1
        # else:
        #     return -1
        return 1
    def calcobs(self,statevector):
        #这一次是保持0位，所以无需处置,但是，不能够一致输出
        #这个函数总是需要考虑到statevector最后一位是标志位的情况

        new_vec=statevector
        # new_vec=self.normalizer.normalize_state(new_vec)
        # 摆角normalized到-pi到pi
        # 考虑到数字范围都不大，在-5到5之间，我想应该不用norm吧....
        return  new_vec[0:4,:]
    def Info_extract(self,statevector):
        self.steps+=1
        if(self.steps>500):
            info="truncated"
            self.steps=0
        else:
            info="notdone"
        return info
    def missiondonejudge(self,statevector):
        done=0
        if(abs(statevector[2,0])>0.2):
            done=1
            self.steps=0
            return done
        if(abs(statevector[0,0])>5):#step步数最好不要大于采样数，因为DQN是从终点开始学起的，想办法加大终点被采样的概率吧
            done=1
            self.steps=0
            return done
        return done
    def randominit(self):
        initstate=[0,0,0.1,0]
        lastobs=self.setstate(initstate)
        return lastobs