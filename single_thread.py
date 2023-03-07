import math

import numpy as np
import torch
class single_thread():
    def __init__(self,index):
        self.index=index
        self.neibor_list=[]
        self.neibor_num=0
        self.state_k=0
        self.v_self_x=0
        self.v_self_y=0
        self.critic_input_k=0 #numpy array类型，用来打断可追溯性
        self.accumulate=0 #累加可以保持tensor的追溯性
    def set_member(self,actor,critic,estimitor,targetgen):
        self.actor=actor
        self.critic=critic
        self.estimitor=estimitor
        self.target_generator=targetgen
        pass
    def set_plant(self,plant):
        self.plant=plant
    def set_calculator(self,calculator):
        self.calculator=calculator
    def copy_params(self,actor_param,critic_param):
        self.actor.actor.load_state_dict(actor_param)
        self.critic.critic.load_state_dict(critic_param)
    def actor_response(self,vector):
        return self.actor.predict(vector)
    def critic_response(self,vector):
        return self.critic.predict(vector)

    def append_neibor(self,neibor):
        self.neibor_list.append(neibor)
        self.neibor_num+=1
    def clear_neibor(self):
        self.neibor_list=[]
    def calc_target(self):
        self.target_xy,self.pos_oc,self.vel_oc=self.target_generator.get_state(self.index)
        # target_xy, pos_oc, vel_oc=self.target_generator.get_state(self.index)
        return self.target_xy, self.pos_oc, self.vel_oc
    def makeinput(self):
        #这里pos_oc非常有意思：要不要给领导者专家知识呢？
        #分别为领导者位置和目标相对位置
        #下面整合时最好弄成张量
        # [x,y,phi,vel,omega]
        errorxy_i=np.mat("0.0;0.0")
        errorV_i=np.mat("0.0;0.0")
        state_relative_total=np.mat("0.0")
        for neibor in self.neibor_list:#这个index必须是从近到远排列
            errorxy_i=errorxy_i+(self.state_k[0:2,0]-self.target_xy)-(neibor.state_k[0:2,0]-neibor.target_xy)
            errorV_i=errorV_i+(np.vstack((self.v_self_x,self.v_self_y))-np.vstack((neibor.v_self_x,neibor.v_self_y)))
            ##上面是车体自身动力误差，下面是计算相对状态差值
            xryr=self.state_k[0:2,0]-neibor.state_k[0:2,0]
            velr=np.vstack((self.v_self_x,self.v_self_y))-np.vstack((neibor.v_self_x,neibor.v_self_y))
            state_relative_sub=np.vstack((xryr,velr))
            state_relative_total=np.vstack((state_relative_total,state_relative_sub))  #垂直
        state_relative_total=state_relative_total[1:,0]
        errorxy_i=errorxy_i+self.state_k[0:2,0]-self.pos_oc-self.target_xy
        errorV_i=errorV_i+np.vstack((self.v_self_x,self.v_self_y))-self.vel_oc
#x y vx vy phi 这个只是变量，不是状态变量哈

        e1=np.vstack((errorxy_i,errorV_i))
        overall_actor= np.vstack((e1,state_relative_total))
        overall_critic=np.vstack((overall_actor,np.vstack((self.v_self_x,self.v_self_y))))
        self.actor_input_vector=torch.tensor(self.paddingzero(overall_actor.transpose(),12))
        self.critic_input_vector=torch.tensor(self.paddingzero(overall_critic.transpose(),14))

    def getR(self,e1,e2):
        Q=np.eye(12)
        R=100.0-(e1*Q*e1.transpose()-e2*Q*e2.transpose())**2  #R应当越靠近0越高
        self.R=R
        return R
    def parse_x(self,x):
        index_list=[]
        for cols in range(0,4):
            if(x[self.index+3,cols]<0):
                break
            else:
                index_list.append(x[self.index+3,cols])
        return index_list
    def make_input_s(self):
        # [x,y,phi,vel,omega]
        vel_x=self.state_k[3,0]*math.cos(self.state_k[2,0])
        vel_y=self.state_k[3,0]*math.sin(self.state_k[2,0])
        errorxy_i=np.mat("0.0;0.0")
        errorV_i=np.mat("0.0;0.0")
        state_relative_total=np.mat("0.0")
        for neibor in self.neibor_list:#这个index必须是从近到远排列

            errorxy_i=errorxy_i+(self.state_k[0:2,0]-self.target_xy)-(neibor.state_k[0:2,0]-neibor.target_xy)
            errorV_i=errorV_i+(np.vstack((vel_x,vel_y))-np.vstack((neibor.v_self_x,neibor.v_self_y)))
            ##上面是车体自身动力误差，下面是计算相对状态差值
            xryr=self.state_k[0:2,0]-neibor.state_k[0:2,0]
            velr=np.vstack((vel_x,vel_y))-np.vstack((neibor.v_self_x,neibor.v_self_y))
            state_relative_sub=np.vstack((xryr,velr))
            state_relative_total=np.vstack((state_relative_total,state_relative_sub)) #垂直

        state_relative_total=state_relative_total[1:,:]
        errorxy_i=errorxy_i+self.state_k[0:2,0]-self.pos_oc-self.target_xy
        errorV_i=errorV_i+np.vstack((vel_x,vel_y))-self.vel_oc
        #x y vx vy phi 这个只是变量，不是状态变量哈

        e1=np.vstack((errorxy_i,errorV_i))

        overall_actor=np.vstack((e1,state_relative_total))
        overall_critic=np.vstack((overall_actor,np.vstack((self.v_self_x,self.v_self_y))))
        overall_actor=self.paddingzero((overall_actor.transpose()),12)
        overall_critic=self.paddingzero((overall_critic.transpose()),14)

        return overall_actor,overall_critic#actor的输入就是state了也就是e向量overall_critic则是本支的动作向量
        #返回值是tensor
    def accumulate_error(self,rewards,value,valuep1):
        gama=0.7#长期回报折扣率
        self.accumulate+=0.5*((value-torch.tensor(rewards)-gama*valuep1)**2)#然后借助这个反向传播到critic的param就行
        return

    def backward(self):
        ##actor是在每一个步长上都需要反向传播的真是麻烦
        ##我现在几乎可以确定，现有的代码可能训练不了目标神经网络
        ##所以，做完一次循环就马上进入模块化环节吧
        ##然后下一次改进算法，先模块化，再做实际功能！！
        ##和TCP_Manage一样，有局部测试的方法
        ##而不是像现在这样，整体做完了才能够进行测试
        self.accumulate.backward()
        pass
    def paddingzero(self,vector,targetlength):
        length=np.size(vector,1)
        if(length<targetlength):
            newmat=np.mat(np.zeros((1,targetlength)))
            for iter in range(0,length):
                newmat[0,iter]=vector[0,iter]
            return newmat
        else:
            return vector
    def actorbackward(self):
        self.value.backward()
        l_rate=0.001
        for f in self.actor.actor.parameters():
            f.data.sub_(f.grad.data * l_rate)

        pass
