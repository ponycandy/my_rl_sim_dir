from PyTorchTool.Policy_proxy import Policy_Proxy
from actornet_policy import actorNet
class M_actpx(Policy_Proxy):
    def __init__(self):
        super(M_actpx, self).__init__([20],[0]) #首先调用父类的初始化函数进行初始化
        self.actor_=actorNet()
        self.setNet(self.actor_)
    def set_member(self):
        pass
    def response(self,vector):
        output=self.predict(vector)
        return output
    def randresponse(self):
        pass
    def chooseaction(self,num):
        pass