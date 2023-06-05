
import os
import glob
import time
import json
from PTorchEnv.model import Actor_Softmax,Critic_PPO
import ray
import torch
from PPO_Single_process import PPO_Single_Process
from Multiprocessing.PPO_Main_Processor import Main_processor
import PTorchEnv.model

ray.init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("configHumanoid.json", "r", encoding='UTF-8') as f:
    params_dict = json.load(f)
worker_num=params_dict["worker_num"]
with open("configHumanoid.json", "w", encoding='UTF-8') as f:
    json.dump(params_dict,f)
#创建PPo的列表
PPO_list = []
#创建公有的网络：
actorNet = PTorchEnv.model.Actor_Humanoid()
criticM = PTorchEnv.model.Critic_PPO(44,128,1)
init_config={"actorNet":actorNet,
             "criticM":criticM,
             "act_space":"Continuous",
             "action_scale":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             "action_bias":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"action_dim":17,"workernumber":worker_num
             }
my_ppo_processor=Main_processor()
my_ppo_processor.init_all_params(init_config)
for i in range(worker_num):
    agent=PPO_Single_Process.remote()
    agent.init_all_params.remote(init_config)
    ray.get(agent.setenv.remote("configHumanoid.json"))
    PPO_list.append(agent)
    time.sleep(1)
#防止两个纪录时间的logger写到同一个文件里面
#初始化可以复用第一种思路的代码，反正验证正确了
while True:
    return_id_list=[]
    for PPO_agent in PPO_list:
        return_id_list.append(PPO_agent.Train_Once.remote())
    results = ray.get(return_id_list)
    my_ppo_processor.merge_list(results)
    #网络更新
    actor,critic=my_ppo_processor.update()
    #参数同步
    return_id_list = []
    for PPO_agent in PPO_list:
        return_id_list.append(PPO_agent.sync_net_params.remote(actor,critic))
    results = ray.get(return_id_list)