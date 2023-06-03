import os
import glob
import time
from datetime import datetime
from PTorchEnv.PPO_Buffer import PPO_Buffer
from PTorchEnv.PPO_Actor_Proxy import PPO_Actor_Proxy
import torch
from PPO import PPO
class Main_processor():
    def __int__(self):
        pass
    def init_all_params(self,params_dict):
        self.replaybuff = PPO_Buffer()
        self.optimizer = PPO()
        self.worker_num= params_dict['workernumber']
        self.optimizer.set_Replaybuff(self.replaybuff, 0.99, 0.0003, 0.001)

        self.actor_proxy = PPO_Actor_Proxy()
        self.actorNet = params_dict['actorNet']
        self.criticM = params_dict['criticM']
        self.actor_proxy.setNet(self.actorNet, self.criticM)
        self.actor_proxy.set_action_dim(params_dict['action_dim'])
        self.actor_proxy.setActFlag(params_dict['act_space'])
        self.actor_proxy.set_range(params_dict['action_scale'], params_dict['action_bias'])
        self.optimizer.setNet(self.actorNet, self.criticM, self.actor_proxy)
        if "single_buffer_size" in params_dict:
            self.optimizer.updateinterval=self.worker_num*params_dict['single_buffer_size']
        #这里涉及到PPO的参数设置问题
    def update(self):
        self.optimizer.stepdone=0
        self.optimizer.update()
        return self.actorNet,self.criticM
    def merge_list(self,results):
        newbuff_states = []
        newbuff_actions = []
        newbuff_logprobs = []
        newbuff_state_values = []
        newbuff_rewards = []
        newbuff_is_terminals = []
        for buffer in results:
            newbuff_states += buffer.states
            newbuff_actions += buffer.actions
            newbuff_logprobs += buffer.logprobs
            newbuff_state_values += buffer.state_values
            newbuff_rewards += buffer.rewards
            newbuff_is_terminals += buffer.is_terminals
        self.replaybuff.states = newbuff_states
        self.replaybuff.actions = newbuff_actions
        self.replaybuff.logprobs = newbuff_logprobs
        self.replaybuff.state_values = newbuff_state_values
        self.replaybuff.rewards = newbuff_rewards
        self.replaybuff.is_terminals = newbuff_is_terminals