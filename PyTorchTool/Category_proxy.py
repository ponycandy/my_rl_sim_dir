import numpy as np
import torch
import random
import math
from PTorchEnv.Typechecker import TensorTypecheck
from PyTorchTool.FileManager import FileManager
class Category_proxy():
    def __init__(self,choise_num):
        self.actions=choise_num
        self.filemanage=FileManager()
        #default epsilon policy
        self.use_eps_flag=0
        self.learning_rate=0
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.eps_step=0
        self.device=0
        self.env=0
        self.random_action_happen_record=0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps_threshold_record=0
        pass
    def set_action_num(self,choise_num):
        self.actions=choise_num
    def predict(self,vector):#type and size has been checked?
        vector=TensorTypecheck(vector).to(torch.float32)
        vector=vector.to(self.device)
        if self.use_eps_flag==0:
            action=self.actor(vector).max(1)[1].view(1, 1)#返回value值最大的那一项神经元对应的index
            index=action
            return index
        else:
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) *math.exp(-1. * self.eps_step / self.EPS_DECAY)
            self.eps_threshold_record=eps_threshold

            self.eps_step += 1
            if sample > eps_threshold:
                # with torch.no_grad():
                # 呃，这个不一定，我的建议是把这个无梯度操作放在外面，所以，with no grad是比detach更好用的方法
                return self.actor(vector).max(1)[1].view(1, 1)
            #为了将
            else:
                # temperory_test
                self.random_action_happen_record+=1
                # return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.int64)  #这个确实是均匀分布
                return torch.tensor([[random.randint(0,self.actions-1)]], device=self.device, dtype=torch.int64)  #这个确实是均匀分布

    def setNet(self,actorNet):
        self.actor=actorNet
    def set_epsilon(self,num):
        self.use_eps_flag=num
    def set_env(self,envnow):
        self.env=envnow
    def random_action(self):#提供外部一个使用完全随机运动的接口
        return random.randint(0,self.actions-1)
    def save_now(self,name):
        torch.save( self.actor, name+'.pt')