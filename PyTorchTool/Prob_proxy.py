from PTorchEnv.Typechecker import TensorTypecheck
import random
import math
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
class Prob_Proxy():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step=1
        self.decayInterval=200000
        self.act_flag=0
        self.action=[0,0,0,0]
        self.use_eps_flag=0
        self.decay_Rate=0.995
    def setNet(self,actor,critic):
        self.actor=actor.to(self.device)
        self.critic=critic.to(self.device)
    def continuous_response(self,vector):
        self.step+=1
        if self.step%self.decayInterval==0:
            self.cov_decay()
            self.step=1
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(vector)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        state_val = self.critic(vector)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        self.action[0]=action  #动作值
        self.action[1]=log_prob  #动作概率值
        self.action[2]=state_val  #估计所得的价值
        self.action[3]=dist#分布
        return self.scale_action(action)
    def Discrete_response(self,vector):
        action_probs = self.actor(vector)
        dist = Categorical(action_probs)
        action = dist.sample()
        state_val = self.critic(vector)
        action_logprob = dist.log_prob(action)

        self.action[0]=action  #动作值
        self.action[1]=action_logprob  #动作概率值
        self.action[2]=state_val  #概率分布函数
        self.action[3]=dist #分布
        return self.chooseaction_pre(action)
    def predict(self,vector):
        vector=TensorTypecheck(vector).to(torch.float32).to(self.device)
        with torch.no_grad():
            if self.act_flag==1:#连续

                act=self.continuous_response(vector)
                return act
            else:
                act=self.Discrete_response(vector)
                return act
    def chooseaction_pre(self,action):
        if(action.shape[0]>1):
            return -1
        else:
            return self.chooseaction(action)

    def set_action_dim(self,dim):

        self.act_dim=dim
        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

    def set_epsilon(self,num):
        self.use_eps_flag=num
    def set_env(self,envnow):
        self.env=envnow

    def set_range(self,scale_range,bias_range):
        self.scale=TensorTypecheck(scale_range).to(self.device)
        self.bias_range=TensorTypecheck(bias_range).to(self.device)


        self.upper_bound = [i + j for i, j in zip(self.scale, self.bias_range)]
        self.lower_bound = [-i + j for i, j in zip(self.scale, self.bias_range)]

        self.lower_ib=-np.mat(np.ones((1,len(self.scale))))
        self.upper_ib=np.mat(np.ones((1,len(self.scale))))
        self.actiodim=len(self.scale)

    def setActFlag(self,status):
        if status=="Discrete":
            self.act_flag=0
        else:
            self.act_flag=1
    def scale_action(self,act):

        real_action=act*self.scale+self.bias_range
        real_action=torch.clamp(real_action,torch.cat( self.lower_bound ), torch.cat(self.upper_bound))

        return real_action.cpu()
    def cov_decay(self):

        self.cov_mat = self.decay_Rate*self.cov_mat
        #与标准不一样，我们依然采用乘法差，减小不必要的负值和最小值判定
