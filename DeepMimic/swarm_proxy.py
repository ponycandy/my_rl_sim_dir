
from PyTorchTool.Prob_proxy import Prob_Proxy
import torch
class SwarmProxy(Prob_Proxy):
    def __init__(self):
        super(SwarmProxy, self).__init__() #首先调用父类的初始化函数进行初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def response(self,vector):
        output=self.predict(vector)

        return self.factor_result(output)
    def factor_result(self,output):
        a=torch.tensor([[output[0,0],output[0,1]],
                        [output[0,2],output[0,3]],
                        [output[0,4],output[0,5]]])
        return a

