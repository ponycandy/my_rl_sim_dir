import torch
from TorchTool import File_manage
class actorNet(torch.nn.Module,File_manage):#这里要求的可变性较小
    def __init__(self):
        super(actorNet, self).__init__() #首先调用父类的初始化函数进行初始化
        self.inputshape=2
        self.writer=2
        self.save_output_layer1=0
        self.save_output_layer2=0
        self.save_output_layer3=0
        self.save_output_layer4=0
        self.save_output_layer5=0
        self.save_output_layer6=0
        D_in=self.inputshape
        Hidden_1=24
        Hidden_2=24
        Hidden_3=24
        D_out=2
        self.layer1 = torch.nn.Linear(D_in, Hidden_1)
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Linear(Hidden_1, Hidden_2)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Linear(Hidden_2, Hidden_3)
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.Linear(Hidden_3, D_out)
        # self.layerx = torch.nn.Linear(D_in, D_out)
        #there is a big problem about training sigmoid function please read:
        #https://stackoverflow.com/questions/73071399/how-to-bound-the-output-of-a-layer-in-pytorch
        #there are a extra layer for nomalize the output to desired range
        self.inputshape=D_in

    # 定义前向传播
    def forward(self, x):
        #先看actionNet的梯度计算吧
        x = x.to(torch.float32)
        x=self.layer1(x)
        # self.save_output_layer1=x
        x=self.layer2(x)
        # self.save_output_layer2=x
        x=self.layer3(x)
        # self.save_output_layer3=x
        x=self.layer4(x)
        # self.save_output_layer4=x
        x=self.layer5(x)
        # self.save_output_layer5=x
        x=self.layer6(x)
        # self.save_output_layer6=x
        x=self.layer7(x)
        self.save_output_layer6=x
        return x

    def get_input_shape(self):
        return self.inputshape
    def clean_grad(self):
        self.zero_grad()
    def my_write_Data(self):
        return