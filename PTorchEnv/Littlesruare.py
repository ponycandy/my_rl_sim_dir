
from TCPenv import TCPenv
class Littlesruare(TCPenv):
    def __init__(self,port,IP):
        super(Littlesruare, self).__init__(port,IP)
        # self.set_pointee(self)
        self.set_simer(self)

    def getreward(self,state,action):
        return 0
    def calcobs(self,statevector):
        #这一次是保持0位，所以无需处置,但是，不能够一致输出
        #这个函数总是需要考虑到statevector最后一位是标志位的情况
        return  statevector[0:1,:]
# getreward(statenext,actionin)
# Info_extract(self.statenext)
# missiondonejudge(self.statenext)