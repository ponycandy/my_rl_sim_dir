import torch

from TCP_Manage import *
import numpy as np
class plant_manager():
    def __init__(self):
        self.Matdatapointer,self.Matadatamanager=Tcpcreator.create_proxy('127.0.0.1',8001)
        self.Matdatapointer1,self.Matadatamanager1=Tcpcreator.create_proxy('127.0.0.1',8002)
        self.matlist=np.mat("0.0,0.0,0.0,0.0,0.0")  # [x,y,phi,vel,omega]其中vel和omega不太必要
    def set_member(self):
        pass
    def add_state_control(self,state,control):
        # self.matlist.append(item)
        # mat = numpy.array(mylist)
        #考虑到这两个都是tensor，建议复制然后转化然后赋值
        state_copy=np.mat("0.0,0.0,0.0,0.0,0.0")
        control_copy=control.detach().numpy()

        state_copy[0,0]=state[0,0]
        state_copy[0,1]=state[1,0]
        state_copy[0,2]=state[2,0]
        state_copy[0,3]=control_copy[0,0]
        state_copy[0,4]=control_copy[0,1]

        self.matlist=np.vstack((self.matlist,state_copy))
        # np.mat(self.matlist)
        # self.stateMat_input= np.vstack((self.stateMat_input, mid_mat.transpose()))
        # pass
    def step_in_plant(self):
        self.stateMat_input=self.matlist[1:,:]  # [x,y,phi,vel,omega]*n
        #首先，这里要发送储存的state--act数据
        self.Matadatamanager.sendMat(self.stateMat_input) #这里没关系，输入的不是tensor或者array就行
        self.Matdatapointer.is_pudated()
        #无需作返回确认，数据会存在缓存区，直到客户端去处理
        cmd = np.mat("1")
        self.Matadatamanager.sendMat(cmd)
        if(self.Matdatapointer.is_pudated()):
            self.stateMat_plant=self.Matdatapointer.getdata()
        return self.stateMat_plant
    def step_in_healthy(self):
        #首先，这里要发送储存的state--act数据
        self.stateMat_input=self.matlist[1:,:]  # [x,y,phi,vel,omega]*n
        self.Matadatamanager1.sendMat(self.stateMat_input)
        self.Matdatapointer1.is_pudated()
        #无需作返回确认，数据会存在缓存区，直到客户端去处理
        cmd = np.mat("0")
        self.Matadatamanager1.sendMat(cmd)
        if(self.Matdatapointer1.is_pudated()):
            self.stateMat_healthy_plant=self.Matdatapointer1.getdata()
        return self.stateMat_healthy_plant
    def clear_mat(self):
        self.matlist=np.mat("0,0,0,0,0")  # [x,y,phi,vel,omega]其中vel和omega不太必要

# 从py到cpp: 3*5矩阵
# [x,y,phi,vel,omega]*3
#其中vel和omega都是

# cpp向pyhton:
# [x,y,phi,vel,omega]*3
# 补上邻居矩阵信息,应该只要index就够了
# [x,x,-1,-1,-1]
# [x,x,-1,-1,-1]
# [x,x,-1,-1,-1]
# 前面两个就是按照从近到远顺序排列的neibor了
# 最后三个-1是补位用的
# 当然，也有可能，x就是-1，那就代表邻居不存在了
#
# n*5矩阵，
#n为agent数目，可以读维度读出来

#stepin指令：1X1矩阵，1表示步进plant,0表示步进healthy
#回波信号直接使用is_updated即可

#不行，需要重订通讯协议，neibor的判断应该使用cpp的既有代码