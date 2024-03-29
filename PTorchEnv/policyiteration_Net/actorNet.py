import torch
from TorchTool import File_manage
class actorNet(torch.nn.Module,File_manage):#这里要求的可变性较小
    def __init__(self,D_in,H,D_out):
        super(actorNet, self).__init__() #首先调用父类的初始化函数进行初始化
        self.inputshape=1
        D_in=self.inputshape
        Hidden_1=10
        Hidden_2=10
        D_out=2
        self.layer1 = torch.nn.Linear(D_in, Hidden_1)
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Linear(Hidden_1, Hidden_2)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Linear(Hidden_2, D_out)
        self.layer6 = torch.nn.Softmax(dim=0)
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
    def normlize(self,x):
        vel_high=1.5
        vel_low=-1.5
        omega_high=1
        omega_low=-1
        y=torch.rand(1,2)
        y[0,0] = (vel_high-vel_low)*x[0,0]+vel_low   #这一步？
        y[0,1] = (omega_high-omega_low)*x[0,1]+omega_low
        return y
    #最后一层使用tensor相乘的办法以获得导数

    def get_input_shape(self):
        return self.inputshape
    def clean_grad(self):
        self.zero_grad()