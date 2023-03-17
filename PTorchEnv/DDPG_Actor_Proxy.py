import copy
from PyTorchTool.Policy_proxy import Policy_Proxy
import torch
class DDPG_Actor_Proxy(Policy_Proxy):
    def __init__(self):
        super(DDPG_Actor_Proxy, self).__init__() #首先调用父类的初始化函数进行初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def setNet(self,Net):
        self.actor=Net
    def deepCopy(self):
        return copy.deepcopy(self)
    def response(self,vector):
        output=self.predict(vector)
        return output.cpu()
    def randresponse(self):
        output=self.random_action()
        self.action=output
        return self.chooseaction(output)