import os
import glob
import time
from datetime import datetime
from PTorchEnv.PPO_Buffer import PPO_Buffer
from PTorchEnv.PPO_Actor_Proxy import PPO_Actor_Proxy
from PTorchEnv.model import Actor_Softmax,Critic_PPO,Actor
from PTorchEnv.Typechecker import TensorTypecheck
import ray
from PPO import PPO
from PyTorchTool.Boardlogger import Boardlogger

@ray.remote
class PPO_Single_Process():
    def __int__(self):

        #最好通过文本实现上面的网络初始化，现阶段暂时使用代码来指定网络
        #下面这些东西的初始化都需要外部完成，所以强烈建议使用文本来初始化
        #
        pass
    def init_all_params(self,params_dict):
        self.rllogger=Boardlogger()

        self.replaybuff = PPO_Buffer()
        self.optimizer = PPO()
        self.optimizer.set_Replaybuff(self.replaybuff, 0.99, 0.0003, 0.001)

        self.actor_proxy = PPO_Actor_Proxy()
        self.actorNet=params_dict['actorNet']
        self.criticM=params_dict['criticM']
        self.actor_proxy.setNet(self.actorNet, self.criticM)
        self.actor_proxy.set_action_dim(params_dict['action_dim'])
        self.actor_proxy.setActFlag(params_dict['act_space'])
        self.actor_proxy.set_range(params_dict['action_scale'],params_dict['action_bias'])
        self.optimizer.setNet(self.actorNet, self.criticM, self.actor_proxy)
        if "single_buffer_size" in params_dict:
            self.optimizer.updateinterval=params_dict['single_buffer_size']
    def setenv(self,env):
        self.env=env  #将会复制所有传入的参数，而不是使用reference!
    def get_act_net(self):
        return self.actorNet
    def get_cri_net(self):
        return self.criticM
    def sync_net_params(self,actnet,criticnet):
        self.actor_proxy.actor.load_state_dict(actnet.state_dict())
        self.actor_proxy.critic.load_state_dict(criticnet.state_dict())
        return 1
    def Train_Once(self):
        #没有optimize环节了，所以务必手动的clear所有内容
        self.replaybuff.clear()
        lastobs = self.env.randominit()
        step=1
        current_ep_reward = 0
        while True:
            action = self.actor_proxy.response(lastobs)
            obs,reward,done,info= self.env.step(action)
            self.replaybuff.appendnew(lastobs, self.actor_proxy.action, obs, reward)
            lastobs = TensorTypecheck(obs)
            if done or info== "truncated":
                self.replaybuff.ResetNotify()  # only notify once!!
                lastobs = self.env.randominit()
                self.rllogger.log_per_step_scalr(current_ep_reward, "ep_reward")
                current_ep_reward = 0
            current_ep_reward += reward
            step += 1
            if(step>self.optimizer.updateinterval):
                break

        return self.replaybuff
    #基本的思路是，大部分的内容不加改变，只是返回值变成所有的经验池而不是优化的网络参数