from PyTorchTool.Category_proxy import Category_proxy
import torch
import copy
class DQN_Actor_Proxy(Category_proxy):
    def __init__(self):
        super(DQN_Actor_Proxy, self).__init__(0) #首先调用父类的初始化函数进行初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def response(self,vector):
        vector=vector.to(torch.float32)
        output=self.predict(vector)
        self.action=output
        return self.chooseaction(output)
    def randresponse(self):
        # 不要在此处更改self.action这个值已经在policy_proxy里面赋值了
        output=self.random_action()
        # self.action=output,
        return output
    def setoutputList(self,actionlst):
        self.actionoutputlist=actionlst
    def chooseaction(self,num):
        return self.actionoutputlist[num]
    def deepCopy(self):
        return copy.deepcopy(self)
