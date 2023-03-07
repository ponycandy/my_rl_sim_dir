import torch
class Tensorparser():
    def __init__(self):
        pass
    def parse(self,tensorm):

        tensorm=tensorm.tolist()
        return tensorm  #a list,However seriliazation is not done here
    def make(self,tensorlist): #输入为完成反序列化的list
        tensorm=torch.tensor(tensorlist)
        return tensorm #输出为正常张量