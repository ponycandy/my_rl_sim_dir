from PTorchEnv.Typechecker import TensorTypecheck
import random
import math
import numpy as np
import torch
class Policy_Proxy():
    def __init__(self,scale_range,bias_range):
        #scale_range:每个维度上下界限范围除以2；bias_range：每个维度的中间位置值
        self.scale=TensorTypecheck(scale_range)
        self.bias_range=TensorTypecheck(bias_range)


        self.upper_bound = [i + j for i, j in zip(self.scale, self.bias_range)]
        self.lower_bound = [-i + j for i, j in zip(self.scale, self.bias_range)]

        self.lower_ib=-np.mat(np.ones((1,len(self.scale))))
        self.upper_ib=np.mat(np.ones((1,len(self.scale))))
        self.actiodim=len(self.scale)
        #default epsilon policy
        self.use_eps_flag=0
        self.EPS_START = 0.99
        self.EPS_END = 0.05
        self.EPS_DECAY = 10000
        self.eps_step=0
        self.device=0
        self.noise_varriance_factor=0.01
        self.varriance=self.noise_varriance_factor*np.eye(self.actiodim) #bydefault
        self.env=0
    def predict(self,vector):
        with torch.no_grad():
            vector=TensorTypecheck(vector)
            if self.use_eps_flag==0:
                action=self.actor(vector)#直接返回动作值
                self.action=action
                act=action*self.scale.t()+self.bias_range.t()
                return act
            else:
                sample = random.random()
                eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *math.exp(-1. * self.eps_step / self.EPS_DECAY)
                self.eps_step += 1
                if sample > eps_threshold:
                    # with torch.no_grad():
                    # 呃，这个不一定，我的建议是把这个无梯度操作放在外面，所以，with no grad是比detach更好用的方法
                    action=self.actor(vector)#直接返回动作值
                    self.action=action
                    act=action*self.scale.t()+self.bias_range.t()
                    return act
                else:
                    #下面是一种添加噪声的手段，也可以使用其它探索策略
                    action_1=self.actor(vector)
                    action_1=action_1.tolist()
                    action_mean = sum(action_1,[])
                    act_array=np.random.multivariate_normal(action_mean, self.varriance)
                    noisy_act= np.clip(act_array, self.lower_ib,self.upper_ib)
                    self.action=noisy_act
                    real_action=torch.tensor(noisy_act)*self.scale.t()+self.bias_range.t()
                    return TensorTypecheck(real_action)
    def setNet(self,actorNet):
        self.actor=actorNet
    def set_epsilon(self,num):
        self.use_eps_flag=num
    def set_env(self,envnow):
        self.env=envnow
    def random_action(self):#提供外部一个使用完全随机运动的接口
        return random.randint(0,self.actions-1)