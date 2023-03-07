import torch
import collections
from collections import namedtuple, deque
from PTorchEnv.ReplayMemory import ReplayMemory
from actorNet import actorNet
import torch.optim as optim
from PTorchEnv.Category_Func import Calc_state_value
m_act_Net=actorNet(2,2,2)
targetNet=actorNet(2,2,2)
BATCH_SIZE=5
GAMMA=0.99
LR=1e-4
memory=ReplayMemory(BATCH_SIZE)
memory.appendnew(1,0,3,4)
memory.appendnew(5,0,7,8)
memory.appendnew(9,1,None,12)
memory.appendnew(13,0,15,16)
memory.appendnew(17,1,19,20)
# buff.push(last_observation, action, observation, reward)
optimizer = optim.AdamW(m_act_Net.parameters(), lr=LR, amsgrad=True)
optimizer.zero_grad()
#优化器是和过程解耦的，其输入只有模型本身的梯度和parameters以及学习速率


state_batch,action_batch,reward_batch,non_final_mask,non_final_next_states=memory.get_Batch_data(BATCH_SIZE)

state_action_values=Calc_state_value(m_act_Net,state_batch,action_batch)

next_state_values = torch.zeros(BATCH_SIZE)
with torch.no_grad():
#     with torch.no_grad的作用
# 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
    next_state_values[non_final_mask] = targetNet(non_final_next_states).max(1)[0]
    #这种赋值方法就是，使用一个tensor来赋值，这个tensor是bool的，在对应true的地方，将值拷贝过来，False地方，拷贝0值
    #例如：non_final_mask=[ True,  True, False,  True,  True]
        #上式右侧为[0.5126, 0.5458, 0.6562, 0.7063]
        #则左侧为[0.5126, 0.5458, 0.0000, 0.6562, 0.7063]
expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch
#1d张量无法转置！！！请先升维


# loss=0.5*(state_action_values-expected_state_action_values)**2
#求出每一个TD误差后，要对总误差求和,也就是上面的向量各项求和，可以使用torch的内建函数解决：
criterion = torch.nn.SmoothL1Loss()
loss = criterion(state_action_values, expected_state_action_values)
loss.backward()
#如何进行参数更新？
#Ok，下面是难点了，我们怎么对神经网络的参数进行更改，我们偏向于手动方法:


# for param in m_act_Net.parameters():
#     param.data = param.data-param.grad*LR
# m_act_Net.zero_grad()

# 但是自动方法也是需要的，先看自动方法吧：

torch.nn.utils.clip_grad_value_(m_act_Net.parameters(), 100) #梯度裁剪，一种防止梯度爆炸的优化策略，非必要
optimizer.step()

#这一步是在计算r+gama*value，也就是贝尔曼公式反推t时刻的价值估计值
print("decode")