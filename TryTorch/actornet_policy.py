import torch
from TorchTool import File_manage
#scene:1st order block pushing
class actorNet(torch.nn.Module,File_manage):#这里要求的可变性较小
    def __init__(self):
        super(actorNet, self).__init__() #首先调用父类的初始化函数进行初始化
        self.inputshape=2
        D_in=self.inputshape
        Hidden_1=8
        Hidden_2=4
        D_out=1
        self.layer1 = torch.nn.Linear(D_in, Hidden_1)
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Linear(Hidden_1, Hidden_2)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Linear(Hidden_2, D_out)
        self.layer6 = torch.nn.Tanh()
        #there is a big problem about training sigmoid function please read:
        #https://stackoverflow.com/questions/73071399/how-to-bound-the-output-of-a-layer-in-pytorch
        #there are a extra layer for nomalize the output to desired range
        self.inputshape=D_in

    # 定义前向传播
    def forward(self, x):
        #先看actionNet的梯度计算吧
        x = x.to(torch.float32)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.layer6(x)

        return x

    def get_input_shape(self):
        return self.inputshape
    def clean_grad(self):
        self.zero_grad()