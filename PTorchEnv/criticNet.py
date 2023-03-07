import torch
class criticNet(torch.nn.Module):#这里要求的可变性较小
    def __init__(self,D_in,H,D_out):
        super(criticNet, self).__init__() #首先调用父类的初始化函数进行初始化
        self.inputshape=14
        D_in=self.inputshape
        Hidden_1=10
        Hidden_2=10
        D_out=1
        self.layer1 = torch.nn.Linear(D_in, Hidden_1)
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Linear(Hidden_1, Hidden_2)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Linear(Hidden_2, D_out)
        #there are a layer5 for nomalize the output to desired range
        self.inputshape=D_in

    # 定义前向传播
    def forward(self, x):
        x = x.to(torch.float32)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        return x
    #可以的，这一路梯度计算没有问题的
    #注意，这里的x不要设置为可追踪的！！！会消耗很多性能！！

    def get_input_shape(self):
        return self.inputshape
    def clean_out_grad(self):
        self.zero_grad()
