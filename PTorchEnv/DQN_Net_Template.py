import torch
from TorchTool import File_manage
class DQN_Net_Template(torch.nn.Module):#这里要求的可变性较小
    def __init__(self):
        super(DQN_Net_Template, self).__init__() #首先调用父类的初始化函数进行初始化

        self.layers=0
    def set_layers(self,layerlist):
        self.layers=layerlist
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