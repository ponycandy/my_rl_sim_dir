import torch
class AC_Critic_Template(torch.nn.Module):#这里要求的可变性较小
    def __init__(self,action_channel,state_channel,output_channel):
        super(AC_Critic_Template, self).__init__() #首先调用父类的初始化函数进行初始化
        self.len_act=len(action_channel)
        self.act_list=[]
        self.state_list=[]
        self.output_list=[]
        for i in range(0,self.len_act):
            exec("self.act_layer{}".format(i)+"=action_channel[i]")
            exec("self.act_list.append("+"self.act_layer{}".format(i)+")")
        self.len_state=len(state_channel)
        for i in range(0,self.len_state):
            exec("self.state_layer{}".format(i)+"=state_channel[i]")
            exec("self.state_list.append("+"self.state_layer{}".format(i)+")")
        self.len_output=len(output_channel)
        for i in range(0,self.len_output):
            exec("self.output_layer{}".format(i)+"=output_channel[i]")
            exec("self.output_list.append("+"self.output_layer{}".format(i)+")")

    # 定义前向传播
    def forward(self,state,action):
        for i in range(0,self.len_act):
            layer=self.act_list[i]
            action=layer(action)
        for i in range(0,self.len_state):
            layer=self.state_list[i]
            state=layer(state)
        x=state+action
        for i in range(0,self.len_output):
            layer=self.output_list[i]
            x=layer(x)
        return x

