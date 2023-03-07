import torch
class TwoLayerNet(torch.nn.Module):#这里要求的可变性较小
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.liner1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.inputshape=D_in
    # 定义前向传播
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    def get_input_shape(self):
        return self.inputshape
