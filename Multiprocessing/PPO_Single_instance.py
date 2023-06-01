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
        self.env = []
        self.replaybuff = PPO_Buffer()
        self.optimizer = PPO()
        self.optimizer.set_Replaybuff(self.replaybuff, 0.99, 0.0003, 0.001)

        self.actor_proxy = PPO_Actor_Proxy()
        self.actorNet = []
        self.criticM = []
        #最好通过文本实现上面的网络初始化，现阶段暂时使用代码来指定网络
        #下面这些东西的初始化都需要外部完成，所以强烈建议使用文本来初始化
        # actor_proxy.setNet(actorNet, criticM)
        # actor_proxy.set_action_dim(2)
        # actor_proxy.setActFlag("Discrete")
        # actor_proxy.set_range([1, 1], [0, 0])
        # optimizer.setNet(actorNet, criticM, actor_proxy)
        #
        # time_step = 0
        #
        # state = env.reset()
        # lastobs = TensorTypecheck(state[0])
        # current_ep_reward = 0
        # epoch = 1
        # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        # writer = SummaryWriter("./my_log_dir/" + TIMESTAMP)
        # start = time.time()
    def setenv(self,env):
        self.env=env
    def get_act_net(self):
        return 1
    def get_cri_net(self):
        return 1
    def sync_net_params(self,actnet,criticnet):
        self.actor_proxy.actor.load_state_dict(actnet.state_dict())
        self.actor_proxy.critic.load_state_dict(criticnet.state_dict())
    def Train_Once(self):
        #这个函数会过一次PPO优化过程，然后返回一个flag值
        lastobs=self.reset_env()
        step=0
        while True:
            action = self.actor_proxy.response(lastobs)
            #这里有一个问题，lastobs的选取没有规范化，我自己的环境和
            #gym的环境选取lastobs的方法是不一样的，解决方法是把reset单独写一个函数
            #保证以后可以更改
            #但是这个方法很笨，统一环境也很笨.....
            #暂时使用单独函数封装来解决问题吧

            #下面同样有这个问题，官方环境和自己环境的返回值不一样，
            #总不能每次都修改吧，但是目前暂时先这样，我还没想好
            #比较通用的封装
            obs, reward, done, truncated, _ = self.env.step(action)
            step+=1
            self.replaybuff.appendnew(lastobs, self.actor_proxy.action, obs, reward)
            lastobs = TensorTypecheck(obs)
            if done or truncated:
                self.replaybuff.ResetNotify()  # only notify once!!
                state = self.env.reset()
                lastobs = TensorTypecheck(state[0])
            self.optimizer.update()
            if(step>self.optimizer.updateinterval):
                break
        return 1
    def reset_env(self):
        return 1