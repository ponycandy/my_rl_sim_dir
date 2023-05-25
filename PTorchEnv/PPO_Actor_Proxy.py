
from PyTorchTool.Prob_proxy import Prob_Proxy
import torch
class PPO_Actor_Proxy(Prob_Proxy):
    def __init__(self):
        super(PPO_Actor_Proxy, self).__init__() #首先调用父类的初始化函数进行初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def response(self,vector):
        output=self.predict(vector)

        return output
    def chooseaction(self,num):  #对于离散状态来说，连续状态选择set range即可
        # if num==0:
        #     #左移
        #     force=0
        #     return force
        # if num==1:
        #     force=1
        #     return force
        return num.item()
