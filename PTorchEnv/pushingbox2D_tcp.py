from TCPenv import TCPenv
class PushingBox2DTCP(TCPenv):
    def __init__(self,port,IP):
        super(PushingBox2DTCP, self).__init__(port,IP)
        # self.set_pointee(self)
        self.set_simer(self)
        self.steps=0

    def getreward(self,state,action):
        if(abs(state[0,0])<0.5 and abs(state[2,0])<0.5):
            reward=1
        else:
            reward=0
        return reward
    def calcobs(self,statevector):
        # statevector[0,0]=statevector[0,0]/10 #大概的范围在5以内，所以正规化一下
        return  statevector
    def Info_extract(self,statevector):
        self.steps+=1
        if(self.steps>200):
            info="truncated"
            self.steps=0
        else:
            info="notdone"
        return info
    def missiondonejudge(self,statevector):
        done=0
        if(abs(statevector[0,0])>5 and abs(statevector[1,0])>5):
            done=1
            self.steps=0
            return done
        return done
    # 仿照penduleum的奖励
    def randonsample(self):
        pass