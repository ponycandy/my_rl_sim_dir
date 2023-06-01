import torch

from PPO_Single_instance import PPO_Single_instance
from PTorchEnv.model import Actor_Softmax,Critic_PPO,Actor
import gym
import ray

ray.init()

worker_num=2

#创建PPo的列表
PPO_list = []
#创建公有的网络：
actorNet = Actor(4, 24, 2)
criticM = Critic_PPO(4, 24, 1)
init_config={"actorNet":actorNet,
             "criticM":criticM,
             "act":"Discrete",
             "action_scale":[1, 1],
             "action_bias":[0, 0],"action_dim":2
             }
#环境变量是自复制的，所以一般来说不通过上面的文本赋值，这样才能够保证TCP的兼容性
#各个进程应该是获取一份copy而不是指针
for i in range(worker_num):
    agent=PPO_Single_instance.remote()
    agent.init_all_params.remote(init_config)
    envnow = gym.make("CartPole-v1")  #这样子不行，请把CartPole-v1用Pyenv封装
    agent.setenv.remote(envnow)
    PPO_list.append(agent)
#初始化每个PPO线程的参数,这个可有点麻烦,包括环境,网络以及其它
while True:
    return_id_list=[]
    for PPO_agent in PPO_list:
        return_id_list.append(PPO_agent.Train_Once.remote())
        #这个设计模式并不要求远程函数一定有返回值，远程函数只要结束就会触发rayget的回调
    results = ray.get(return_id_list)
    #计算平均的参数值
    act_net_list=[]
    cri_net_list=[]
    for PPO_agent in PPO_list:
        act_net_list.append(ray.get(PPO_agent.get_act_net.remote()))
        cri_net_list.append(ray.get(PPO_agent.get_cri_net.remote()))
    act_model_dict=actorNet.state_dict()
    cri_model_dict=criticM.state_dict()
    for k1,k2 in zip(act_model_dict.keys(),cri_model_dict.keys()):
        act_results=torch.zeros_like(act_model_dict[k1])
        cri_results=torch.zeros_like(cri_model_dict[k2])
        i=0
        for PPO_agent in PPO_list:
            act_results+=act_net_list[i].state_dict()[k1]
            cri_results+=cri_net_list[i].state_dict()[k2]
            i+=1
        act_results/=worker_num
        cri_results/=worker_num
        act_model_dict[k1]=act_results
        cri_model_dict[k2]=cri_results
    actorNet.load_state_dict(act_model_dict)
    criticM.load_state_dict(cri_model_dict)
    #参数平均值已经载入到主进程的网络中，接下来，同步到各个子进程的网络，这一步对主进程只读
    # 不知道会不会出bug...
    return_id_list = []
    for PPO_agent in PPO_list:
        return_id_list.append(PPO_agent.sync_net_params.remote(actorNet,criticM))
    results = ray.get(return_id_list)
        #不使用remote呼出remote的对象，会报错：
        #Actor methods cannot be called directly.
    #一个进程结束，然后重新开始循环
#以上多进程是基于我对ray的特性的理解，暂时没有用到shared object的特性