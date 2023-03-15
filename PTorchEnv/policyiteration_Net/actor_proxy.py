from actorNet import actorNet
from Category_proxy import Category_proxy
class actor_proxy(Category_proxy):
    def __init__(self,actionum):
        self.actor=actorNet(1,11,1)
        super(actor_proxy, self).__init__(actionum) #首先调用父类的初始化函数进行初始化
    def set_member(self):
        pass
    # def predict(self,vector):
    #     output=self.actor.forward(vector)
    #     self.action=output
    #     return self.action