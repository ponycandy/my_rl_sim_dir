import torch
class criticNet(torch.nn.Module):#这里要求的可变性较小
    def __init__(self,D_in,H,D_out):
        super(criticNet, self).__init__() #首先调用父类的初始化函数进行初始化
        state_shape=2
        action_shape=1
        Hidden_1=30
        D_out=1
        self.layer1 = torch.nn.Linear(state_shape, Hidden_1)
        self.layer2 = torch.nn.Linear(action_shape, Hidden_1)
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.Linear(Hidden_1, D_out)
        #there are a layer5 for nomalize the output to desired range
        self.inputshape=D_in

    # 定义前向传播
    def forward(self, state,action):
        state=state.to(torch.float32)
        action=action.to(torch.float32)
        state=self.layer1(state)
        action=self.layer2(action)
        mid=self.layer3(action+state)
        output=self.layer4(mid).to(torch.float32)

        return output
        #可以的，这一路梯度计算没有问题的
    #注意，这里的x不要设置为可追踪的！！！会消耗很多性能！！

    def get_input_shape(self):
        return self.inputshape
    def clean_out_grad(self):
        self.zero_grad()
