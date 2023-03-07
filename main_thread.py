import math

import numpy as np
import torch
class main_thread():
    def __init__(self):
        self.child_list=[]
        self.child_num=0
        self.reward=0

    def update_all_child(self):
        for threads in self.child_list:
            threads.copy_params(self.actor.actor.state_dict(),self.critic.critic.state_dict())
        pass
    def value_add(self):
        staterror_1_list=np.mat("0,0,0,0,0,0,0,0,0,0,0,0")
        critic_input_1_list=np.mat("0,0,0,0,0,0,0,0,0,0,0,0,0,0")
        staterror_2_list=np.mat("0,0,0,0,0,0,0,0,0,0,0,0")
        critic_input_2_list=np.mat("0,0,0,0,0,0,0,0,0,0,0,0,0,0")

        self.update_child_state(self.xrkp1)
        for threads in self.child_list:
            state_error2,critic_input_2=threads.make_input_s() #state=12 critic=14
            staterror_2_list=np.vstack((staterror_2_list,state_error2))
            critic_input_2_list=np.vstack((critic_input_2_list,critic_input_2))
            threads.clear_neibor()

        self.update_child_state(self.xkp1)
        for threads in self.child_list:
            state_error1,critic_input_1=threads.make_input_s() #state=12 critic=14
            staterror_1_list=np.vstack((staterror_1_list,state_error1))
            critic_input_1_list=np.vstack((critic_input_1_list,critic_input_1))




        num=1
        for threads in self.child_list:
            rewards=threads.getR(staterror_1_list[num,:],staterror_2_list[num,:])
            threads.valueplus1=threads.critic_response(torch.tensor(critic_input_1_list[num,:]))
            value=threads.critic_response(threads.critic_input_k)
            #此处必须重新进行预测,以打断计算图
            threads.accumulate_error(rewards,threads.valueplus1,value)
            num+=1

    def update_self(self):
        l_rate=0.000001
        ##self.actor.actor.state_dict()['layer1.weight']此为神经网络的参数访问方式
        for threads in self.child_list:
            # for f in self.actor.actor.parameters():
            #     f.data.sub_(f.grad.data * l_rate)
            for f in self.critc.critc.parameters():
                f.data.sub_(f.grad.data * l_rate)
        threads.actor.actor.zero_grad()
        threads.critc.actor.zero_grad()


    def set_member(self,actor,critic,plant,m_imediate_r):
        self.actor=actor
        self.critic=critic
        self.plant=plant
        self.Imediater=m_imediate_r
        pass
    def Rcircular(self,num,max):
        if num<0:
            return max
        if num>max:
            return 0
        return num
    def distancecalc(self,state1,state2):
        return (state1[0,0]-state2[0,0])**2+(state1[1,0]-state2[1,0])**2
    def init_agents_state(self,info):
# [x,y,phi,vel,omega]
        num=0
        for threads in self.child_list:
            threads.state_k=info[num,:].transpose()
            num+=1
            #排序邻居
        num=0
        for threads in self.child_list:
            state1=self.child_list[self.Rcircular(num-1,2)].state_k
            state2=self.child_list[self.Rcircular(num+1,2)].state_k
            dis1=self.distancecalc(threads.state_k,state1)
            dis2=self.distancecalc(threads.state_k,state2)

            if dis1<=dis2 :
                threads.neibor_list.append(self.child_list[self.Rcircular(num-1,2)])
                threads.neibor_list.append(self.child_list[self.Rcircular(num+1,2)])
            else:
                threads.neibor_list.append(self.child_list[self.Rcircular(num+1,2)])
                threads.neibor_list.append(self.child_list[self.Rcircular(num-1,2)])

            v_self_x=threads.state_k[3,0]*math.cos(threads.state_k[2,0])
            v_self_y=threads.state_k[3,0]*math.sin(threads.state_k[2,0])
            threads.v_self_x=v_self_x
            threads.v_self_y=v_self_y
            num+=1

        #OK那就只返回neiborindex好了
    def add_child_thread(self,child_thread):
        self.child_list.append(child_thread)
        self.child_num+=1
        pass
    def preditc_output(self):
        num=0
        #d
        for threads in self.child_list:
            threads.target_xy,threads.pos_oc,threads.vel_oc=threads.calc_target()
        #需要首先获得各个agent的目标位置和追随速度
        for threads in self.child_list:
            threads.makeinput()
            uk=threads.actor_response(threads.actor_input_vector)#问题在这里,不知道为什么传播不动了
            critic_input_k=torch.cat((threads.actor_input_vector,uk),dim=1)
            #那，考虑单个赋值
            threads.critic_input_k=critic_input_k.detach().numpy()
          #  critic_input_ks=critic_input_k.detach().numpy()
            threads.critic_input_k=torch.tensor(threads.critic_input_k)
            #这里uk是一个tensor，当value发生反向传播的时候
            #我们希望这里能够关联
            #但是当TE发生反向传播的时候，我们又不希望这里发生关联
            threads.value=threads.critic_response(critic_input_k)
            ##此处获取的值用来对actorNet反向传播
            num+=1
        pass
    def step_in(self):
        for threads in self.child_list:
            self.plant.add_state_control(threads.state_k,threads.actor.action)
            threads.clear_neibor()
        self.xrkp1=self.plant.step_in_plant()
        self.xkp1=self.plant.step_in_healthy()
        #呃，这两个的相邻状态又不一样,那就只能够临时解包了

            #这里可以直接加，不影响梯度
            #ohoh不妙，这里需要一个线程自身的状态以及邻居的状态，也就是cpp仿真器里边的那些东东....
            #所以建议这部分回传，重新制定协议
            #woca，好麻烦呐,这个应该是用两个imediate的乘积的差值
            #而且这个必须是使用tensor的了
    def makeup_state(self,index,stateNow):
        return 0  #根据现有全部状态，确定分状态？
    #不好，状态应该保留在single threads里面
    def backward_child(self):
        for threads in self.child_list:
            threads.backward()
    def update_child_state(self,state):
        num=0
        for threads in self.child_list:
            threads.state_k=np.mat(state[num,:]).transpose()
            for agents in range(0,4):
                if state[num+3,agents]<0:
                    break
                else:
                    threads.neibor_list.append(self.child_list[int(state[num+3,agents])])
            num+=1
        pass
    def train_actors(self):
        for threads in self.child_list:
            threads.actorbackward()
        pass