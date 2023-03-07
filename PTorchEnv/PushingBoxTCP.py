from TCPenv import TCPenv
class PushingBoxTCP(TCPenv):
    def __init__(self,port,IP):
        super(PushingBoxTCP, self).__init__(port,IP)
        # self.set_pointee(self)
        self.set_simer(self)
        self.steps=0

    def getreward(self,state,action):
        x=state[0,0]
        if(abs(x)<1):
            return 1
        return 0
    def calcobs(self,statevector):
        return  statevector
    def Info_extract(self,statevector):
        info="fine"
        if(abs(statevector[0,0])>5):
            info="speed_out"
        return info
    def missiondonejudge(self,statevector):
        done=0
        self.steps+=1
        if(abs(statevector[0,0])>5):
            info="speed_out"
            done=1
            self.steps=0
            return done
        if(self.steps>100):
            info="speed_out"
            done=1
            self.steps=0
            return done
        return done
