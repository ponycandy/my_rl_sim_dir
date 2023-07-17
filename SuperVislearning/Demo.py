#这里我们使用监督式训练来训练一个PID的控制器
#然后在refine阶段正式训练
from PTorchEnv.model import Actor_Softmax,Critic_PPO,Actor

from PyTorchTool.FileManager import FileManager

import torch

import torch
import torch.optim as optim
def random_tensor(num):
    with torch.no_grad():
      # generate a 100 * 4 Tensor with random values in range 0 to 1
      tensor = torch.rand(num, 4)
      # scale the tensor to range -5 to 5,排序是
      input = tensor * 10 - 5
      # return the tensor
      x_action = -input[:, 0]-input[:, 1]
      y_action = -input[:, 2] - input[:, 3]
    #控制量验证没有错误
      output=torch.stack((x_action,y_action),dim=1)
      output=output/10 #这样就和Actornet的允许输出范围一致了
      return input,output

num=1000;
Trainiter=4000
actorNet=Actor(4,24,2)
optimizer = optim.AdamW(actorNet.parameters(), lr=0.0001, amsgrad=True)
loss_fn = torch.nn.MSELoss()
loss=100
verify_loss=100
#生成随机训练数据，4入2出

for i in range(Trainiter):
# while verify_loss>0.0854:
    obs, action = random_tensor(num)
    verifyobs, verifyaction = random_tensor(num)
    predicted=actorNet(obs)
    loss = loss_fn(predicted, action)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(actorNet.parameters(), 100)  # 梯度裁剪，一种防止梯度爆炸的优化策略，非必要
    optimizer.step()
    # print("loss is ",loss)
    with torch.no_grad():
        v_predict=actorNet(verifyobs)
        verify_loss = loss_fn(v_predict, verifyaction)
        print("verify_loss is ", verify_loss)



manager = FileManager()
manager.save_model_out(actorNet, "actor_net_pretrained")

print("done")