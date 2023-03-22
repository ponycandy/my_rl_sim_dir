from actor_proxy import actor_proxy
from PTorchEnv.CartpoleTCP import CartpoleTCP
from PTorchEnv.ReplayMemory import ReplayMemory
from PTorchEnv.Prioritized_Replaybuffer import Prioritized_Replaybuffer
from PTorchEnv.DiscreteOpt import DiscreteOpt
import random
from PyTorchTool.RLDebugger import RLDebugger
from PTorchEnv.matrix_copt_tool import deepcopyMat
from tensorboardX import SummaryWriter
from PTorchEnv.RL_parameter_calc import RL_Calculator
from datetime import datetime
import torch
import gymnasium as gym
from PPO import PPO
RL_logger=RL_Calculator()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
# ploterr=RLDebugger()

envnow = gym.make("CartPole-v1")

ppo_agent = PPO(4, 2, 1e-4, 1e-4, 0.9, 80, 0.2, False, 4000,0.6   )
state, info = envnow.reset()

step_done=0
total_reward=0
epoch=1
while True:
    action=ppo_agent.select_action(state)

    obs,reward,done,info,_=envnow.step(action)
    total_reward+=reward

    if done :
        # print("the action is:",action,"angle now:",obs[1,0])

        obs=None


    ppo_agent.buffer.rewards.append(reward)
    ppo_agent.buffer.is_terminals.append(done)
    callback=ppo_agent.update()
    if callback==0:
        pass
    else:
        epoch+=1
        writer.add_scalar("reward",total_reward,epoch)

    step_done+=1


#这样一来，总的代码量就大量减小了
#同时，需要重复的低级细节：如何解包压缩经验池，如何计算目标函数，如何升级权值，就可以被隐藏了
#目前暴露的细节就是：1.我们可以选择合适选取经验池进行训练 2.选择网络升级的方法
#封包的细节：1.off-policy的实施 2.targetNet的实施
#要考虑的是通用性，现有的方法应该能够通用于一切value based agent

#当前模型已经分析有效
#试一下pendulum经典环境，学习非常之快，几乎一个迭代就能够获得目标行为



