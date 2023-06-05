import os
import glob
import time
from datetime import datetime
from PTorchEnv.PPO_Buffer import PPO_Buffer
from PTorchEnv.PPO_Actor_Proxy import PPO_Actor_Proxy
from PTorchEnv.Typechecker import TensorTypecheck
import ray
from PPO import PPO
from PyTorchTool.Boardlogger import Boardlogger
import json
import torch
#注意，此处的进程已经加速了，而且也已经用上了GPU
#通过检测，对应的PID确实运行在了不同的核上面，无法通过下面的num_cpus更改
#因为调度到单个核心是CPU硬件层级的API,故Ray也必然无法操作这个
#这个num_cpus只是给人看的，不代表它真的能控制单个CPU跑单个进程
#想一想我们怎么跑我们自己的代码就知道了，这是做不到的

#所以，想要加速，多机多cpu（不是多核），是必须的
@ray.remote(num_cpus=1)
class PPO_Single_Process():
    def __int__(self):

        #最好通过文本实现上面的网络初始化，现阶段暂时使用代码来指定网络
        #下面这些东西的初始化都需要外部完成，所以强烈建议使用文本来初始化
        #
        pass
    def init_all_params(self,params_dict):

        # with open("config.json", "r", encoding='UTF-8') as f:
        #     params_dict = json.load(f)
        self.rllogger=Boardlogger()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
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
    def setenv(self,configfilename):
        with open(configfilename, "r", encoding='UTF-8') as f:
            params_dict = json.load(f)
        if "extra_cmd_prev" in params_dict:
            for command in params_dict["extra_cmd_prev"]:
                exec(command["cmd"])
        exec(params_dict["env_import"])
        self.env=eval(params_dict["env_cmd"])
        if "extra_cmd_after" in params_dict:
            for command in params_dict["extra_cmd_after"]:
                exec(command["cmd"])
                #理论上需要把所有的python脚本写到这个cmd里面
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