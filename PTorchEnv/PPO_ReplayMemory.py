# implement refernece:
# https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
from collections import namedtuple, deque
import torch
import random
import numpy as np
from PTorchEnv.Typechecker import TensorTypecheck
from PTorchEnv.Typechecker import TensorTypecheck
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class PPO_ReplayMemory(object):

    def __init__(self, capacity):
        self.recordinglength=0
        self.batch_lastobs = []             # batch observations
        self.batch_obs = []             # batch observations
        self.batch_acts = []            # batch actions
        self.batch_log_probs = []       # log probs of each action
        self.batch_rews = []            # batch rewards
        self.batch_rtgs = []            # batch rewards-to-go
        self.batch_lens = []            # episodic lengths in batch
        self.ep_rews = []
        self.steps_count=0
        self.total_timestep=0
    def ResetNotify(self):
        self.batch_rews.append(self.ep_rews)
        self.ep_rews = []  #清空所有奖励，因为这是一个on-policy的算法
        #                     可能有bug，清空后检查一下batch_rews
        self.batch_lens.append(self.steps_count + 1) #需要知道这一次reset总共走了多少步
        self.steps_count=0
        pass
    def appendnew(self,lastobs,act,state,reward):
        # 需要在此处获知episode是否终结的消息
        # 只要进行重置就是终结
        # 重置的条件是terminated或者truncated
        # 这样的话，self.max_timesteps_per_episode就必须大于
        # 游戏规则的truncated极限
        self.total_timestep+=1 #计算到update策略之前的总步数，这个值只能被优化器设0
        self.steps_count+=1
        self.recordinglength+=1
        self.batch_lastobs.append(TensorTypecheck(lastobs))
        self.batch_obs.append(state)
        self.batch_acts.append(act[0])  #神经网络输出的act值
        self.batch_log_probs.append(act[1])  #神经网络输出的act对应log值
        self.ep_rews.append(reward)
    def clearbuffer(self):
        self.batch_lastobs = []             # batch observations
        self.batch_obs = []             # batch observations
        self.batch_acts = []            # batch actions
        self.batch_log_probs = []       # log probs of each action
        self.batch_rews = []            # batch rewards
        self.batch_rtgs = []            # batch rewards-to-go
        self.batch_lens = []            # episodic lengths in batch
        self.ep_rews = []
        self.steps_count=0
        self.total_timestep=0
