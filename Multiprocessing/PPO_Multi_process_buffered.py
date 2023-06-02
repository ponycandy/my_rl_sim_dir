
import os
import glob
import time
from datetime import datetime
from PTorchEnv.PPO_Buffer import PPO_Buffer
from PTorchEnv.PPO_Actor_Proxy import PPO_Actor_Proxy
from PTorchEnv.model import Actor_Softmax,Critic_PPO,Actor
from PTorchEnv.Typechecker import TensorTypecheck
import ray
import torch
from PPO import PPO
from PyTorchTool.Boardlogger import Boardlogger
from PPO_Single_process import PPO_Single_Process
from PTorchEnv.CartpoleGym import CartPoleGym
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

#伪代码：
# while True:
#     collect_buffer()
#     gain_buffer()
#     update_network()
#     sync_model()
#这个还是可以做的....
#我们在主线程中定义一个PPO的类，也就是Main_processor来处理update问题

ray.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

worker_num=4

#创建PPo的列表
PPO_list = []
#创建公有的网络：
actorNet = Actor_Softmax(4, 24, 2)
criticM = Critic_PPO(4, 24, 1)
init_config={"actorNet":actorNet,
             "criticM":criticM,
             "act_space":"Discrete",
             "action_scale":[1, 1],
             "action_bias":[0, 0],"action_dim":2,"workernumber":worker_num
             }
#环境变量是自复制的，所以一般来说不通过上面的文本赋值，这样才能够保证TCP的兼容性
#各个进程应该是获取一份copy而不是指针

my_ppo_processor=Main_processor()
my_ppo_processor.init_all_params(init_config)
for i in range(worker_num):
    agent=PPO_Single_Process.remote()
    agent.init_all_params.remote(init_config)
    envnow=CartPoleGym()
    agent.setenv.remote(envnow)
    PPO_list.append(agent)
    time.sleep(1)
#防止两个纪录时间的logger写到同一个文件里面
#初始化可以复用第一种思路的代码，反正验证正确了
while True:
    return_id_list=[]
    for PPO_agent in PPO_list:
        return_id_list.append(PPO_agent.Train_Once.remote())
        #这个设计模式并不要求远程函数一定有返回值，远程函数只要结束就会触发rayget的回调
    results = ray.get(return_id_list)
    #获得的结果为replaybuff,接下来的问题在于，如何拼到一块儿，粗浅的理解，合并其下四个子列表就行
    # 记得手动设置steps，不然会没办法进到PPO的updaue里面
    newbuff_states=[]
    newbuff_actions=[]
    newbuff_logprobs=[]
    newbuff_state_values=[]
    newbuff_rewards=[]
    newbuff_is_terminals=[]
    for buffer in results:
        newbuff_states+=buffer.states
        newbuff_actions+=buffer.actions
        newbuff_logprobs+=buffer.logprobs
        newbuff_state_values+=buffer.state_values
        newbuff_rewards+=buffer.rewards
        newbuff_is_terminals+=buffer.is_terminals
    my_ppo_processor.replaybuff.states=newbuff_states
    my_ppo_processor.replaybuff.actions=newbuff_actions
    my_ppo_processor.replaybuff.logprobs=newbuff_logprobs
    my_ppo_processor.replaybuff.state_values=newbuff_state_values
    my_ppo_processor.replaybuff.rewards=newbuff_rewards
    my_ppo_processor.replaybuff.is_terminals=newbuff_is_terminals
    #网络更新
    actor,critic=my_ppo_processor.update()
    #参数同步
    return_id_list = []
    for PPO_agent in PPO_list:
        return_id_list.append(PPO_agent.sync_net_params.remote(actor,critic))
    results = ray.get(return_id_list)
#测试对比2 worker差于 1 worker
#但是4 worker显著好于1 worker，可复现
#所以，目前的多线程，我们只能说理论上会快于单线程算法（多线程buffer收集原理上是更快的）
#但是实际表现未知，两种多线程的实现本质上都是加大了同时取样数，所以原理上都会更快
#但是实验上的显著性不高,也可能是环境所致，但根据主流，我们采取此处的多线程方法
#暂时就先不使用调参算法

#下一步，实现SwarmTCP的多环境并行（第二种初始化接口？！）