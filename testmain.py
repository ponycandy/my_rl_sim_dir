from main_thread import main_thread
from single_thread import single_thread
from plant_manager import plant_manager
from Imediate_reward import Imediate_reward
from error_estimate import error_estimate
from actor_proxy import actor_proxy
from critic_proxy import critic_proxy
from TargetGen import TargetGen
import numpy as np

m_mainthread=main_thread()
m_target=TargetGen()
m_plant_manage=plant_manager()

m_imediate_r=Imediate_reward()


def create_a_child_thread(num):
    m_singlethread=single_thread(num)
    m_errorestimate=error_estimate()
    subactor=actor_proxy()
    subcritic=critic_proxy()
    m_singlethread.set_member(subactor,subcritic,m_errorestimate,m_target)
    return m_singlethread

main_actor=actor_proxy()
main_critic=critic_proxy()

m_mainthread.set_member(main_actor,main_critic,m_plant_manage,m_imediate_r)
num_of_child=3
for child_index in range(0,num_of_child):
        a_thread=create_a_child_thread(child_index)
        a_thread.set_plant(m_plant_manage)
        reward_calculator=Imediate_reward()
        a_thread.set_calculator(reward_calculator)
        m_mainthread.add_child_thread(a_thread)

N_loops=200
max_step=100 #30ms/step
initial_state=np.mat("2,1,0,0,0; 2,-1,0,0,0;3,5,0,0,0")
# [x,y,phi,vel,omega]这个只是变量，不是状态变量哈
m_mainthread.init_agents_state(initial_state)
for training_loop in range(0,N_loops):
    m_mainthread.update_all_child()
    #从这里开始训练，这一部分不要写道类里面
    for t in range(0,max_step):
        m_mainthread.preditc_output()  #不要把输入变成可追踪的！！！actor和critic都是
        m_mainthread.step_in()
        # m_mainthread.update_child_state()
        m_target.evolve()     #内部储存值为上一targetxy，外部储存值为下一targetxy,
        #上面这个可以改，必要的话我们也可以在计算中用下一目标的target，这有可能使得系统具有预测性
        m_mainthread.value_add()
        m_mainthread.train_actors()

    m_mainthread.backward_child()
    #从这里开始训练，这一部分不要写到类里面

    m_mainthread.update_self()
    print("loop %d finished",training_loop)




