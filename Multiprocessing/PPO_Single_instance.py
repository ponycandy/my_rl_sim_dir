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

@ray.remote
class PPO_Single_instance():
    def __int__(self):
        self.replaybuff = PPO_Buffer()
        self.optimizer = PPO()
        self.optimizer.set_Replaybuff(self.replaybuff, 0.99, 0.0003, 0.001)

        self.actor_proxy = PPO_Actor_Proxy()
        #最好通过文本实现上面的网络初始化，现阶段暂时使用代码来指定网络
        #下面这些东西的初始化都需要外部完成，所以强烈建议使用文本来初始化
        #
    def init_all_params(self,params_dict):
        self.actorNet=params_dict['actorNet']
        self.criticM=params_dict['criticM']
        self.actor_proxy.setNet(self.actorNet, self.criticM)
        self.actor_proxy.set_action_dim(params_dict['action_dim'])
        self.actor_proxy.setActFlag(params_dict['action_dim'])
        self.actor_proxy.set_range(params_dict['action_scale'],params_dict['action_bias'])
        self.optimizer.setNet(self.actorNet, self.criticM, self.actor_proxy)
    def setenv(self,env):
        self.env=env  #将会复制所有传入的参数，而不是使用reference!
    def get_act_net(self):
        return 1
    def get_cri_net(self):
        return 1
    def sync_net_params(self,actnet,criticnet):
        self.actor_proxy.actor.load_state_dict(actnet.state_dict())
        self.actor_proxy.critic.load_state_dict(criticnet.state_dict())
        return 1
    def Train_Once(self):
        #这个函数会过一次PPO优化过程，然后返回一个flag值
        lastobs = self.env.randominit()
        step=0
        while True:
            action = self.actor_proxy.response(lastobs)
            #这里有一个问题，lastobs的选取没有规范化，我自己的环境和
            #gym的环境选取lastobs的方法是不一样的，解决方法是把reset单独写一个函数
            #保证以后可以更改
            #但是这个方法很笨，统一环境也很笨.....
            #暂时使用单独函数封装来解决问题吧

            #有了，创建一个新的env，这个env按照我自己的标准输出来输出（继承pyenv）
            #这样包一层后再放到这里面来用，API就统一了,这样就不需要额外的reset函数

            #下面同样有这个问题，官方环境和自己环境的返回值不一样，
            #总不能每次都修改吧，但是目前暂时先这样，我还没想好
            #比较通用的封装
            obs,reward,done,info= self.env.step(action)

            self.replaybuff.appendnew(lastobs, self.actor_proxy.action, obs, reward)
            lastobs = TensorTypecheck(obs)
            if done or info== "truncated":
                self.replaybuff.ResetNotify()  # only notify once!!
                lastobs = self.env.randominit()
            self.optimizer.update()
            step += 1
            if(step>self.optimizer.updateinterval):
                break
        return 1