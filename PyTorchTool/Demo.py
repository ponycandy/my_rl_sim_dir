import torch
import time
from FileManager import FileManager
class actorNet(torch.nn.Module,FileManager):#这里要求的可变性较小
    def __init__(self,D_in,H,D_out):
        super(actorNet, self).__init__() #首先调用父类的初始化函数进行初始化
        self.inputshape=12
        D_in=self.inputshape
        Hidden_1=10
        Hidden_2=10
        D_out=2
        self.layer1 = torch.nn.Linear(D_in, Hidden_1)
        self.layer2 = torch.nn.ReLU(inplace=True)
        self.layer3 = torch.nn.Linear(Hidden_1, Hidden_2)
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Linear(Hidden_2, D_out)
        self.layer6 = torch.nn.Sigmoid()
        #there is a big problem about training sigmoid function please read:
        #https://stackoverflow.com/questions/73071399/how-to-bound-the-output-of-a-layer-in-pytorch
        #there are a extra layer for nomalize the output to desired range
        self.inputshape=D_in

    # 定义前向传播
    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.layer6(x)
        x=self.normlize(x)


        return x
    def normlize(self,x):
        vel_high=1.5
        vel_low=-1.5
        omega_high=1
        omega_low=-1
        x[1]=x[1]*(vel_high-vel_low) + vel_low
        x[2]=x[2]*(omega_high-omega_low) + omega_low
        return x
    def get_input_shape(self):
        return self.inputshape

m_test_act=actorNet(3,1,3)
m_test_act.init_file_manage()
m_test_act.set_counter_mode(m_test_act,'tes',10)
while(1):
    m_test_act.counter()
    time.sleep(1)
