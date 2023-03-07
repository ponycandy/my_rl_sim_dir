from TCPenv import TCPenv
class CartpoleTCP(TCPenv):
    def __init__(self,port,IP):
        super(CartpoleTCP, self).__init__(port,IP)
        # self.set_pointee(self)
        self.set_simer(self)
        self.steps=0

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
        phi=statevector[2,0]
        if(phi>3.1415926535):
            phi=2*3.1415926535-phi
        if(phi<-3.1415926535):
            phi=phi+2*3.1415926535
        statevector[2,0]=phi
        return  statevector[0:4,:]
    def Info_extract(self,statevector):
        info="fine"
        # if(abs(statevector[0,0])>5):
        #     info="speed_out"
        return info
    def missiondonejudge(self,statevector):
        done=0
        self.steps+=1
        if(abs(statevector[2,0])>0.2):
            info="speed_out"
            done=1
            self.steps=0
            return done
        if(self.steps>140):#step步数最好不要大于采样数，因为DQN是从终点开始学起的，想办法加大终点被采样的概率吧
            info="speed_out"
            done=1
            self.steps=0
            return done
        if(abs(statevector[0,0])>5):#step步数最好不要大于采样数，因为DQN是从终点开始学起的，想办法加大终点被采样的概率吧
            info="speed_out"
            done=1
            self.steps=0
            return done
        return done
