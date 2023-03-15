from actorNet import actorNet
from PyTorchTool.Category_proxy import Category_proxy
import torch
class actor_proxy(Category_proxy):
    def __init__(self):
        super(actor_proxy, self).__init__(3) #首先调用父类的初始化函数进行初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_=actorNet().to(self.device)
        self.setNet(self.actor_)
        self.writer=0
        pass
    def set_member(self):
        pass
    def response(self,vector):
        output=self.predict(vector)
        self.action=output
        return self.chooseaction(output)
    def randresponse(self):
        output=self.random_action()
        self.action=output
        return self.chooseaction(output)
    def chooseaction(self,num):
        if num==0:
        #左移
            force=100
        if num==1:
            force=-100
        if num==2:
            force=0
        return force