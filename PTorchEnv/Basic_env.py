import numpy as np
import torch
class Basic_env:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def set_pointee(self,pointee):
        self.pointee=pointee
        if(hasattr(self.pointee.pointee, 'calcobs')==False):
            assert False,"calcobs function not define!"
        if(hasattr(self.pointee.pointee, 'getreward')==False):
            assert False,"getreward function not define!"
        if(hasattr(self.pointee.pointee, 'randonsample')==False):
            assert False,"randonsample function not define!"
    def step(self,action): #这个函数必须继承加重写
        actionin=self.typecheck(action)
        statenext,done,info= self.pointee.stepin(actionin)
        obs=self.pointee.pointee.calcobs(statenext)

        obs=self.typecheck(obs)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        reward=self.pointee.pointee.getreward(statenext,actionin)


        return obs,reward,done,info
    def setstate(self,state):
        callbackinfo=self.pointee.setstate(self.typecheck(state))
        return callbackinfo
    def randominit(self):
        initstate=self.pointee.randonsample()
        lastobs=self.pointee.setstate(self.typecheck(initstate))
        return lastobs
    def typecheck(self,vector):##始终使得宽小于长

        if isinstance(vector,list): #按照list类型进行转化
            mat=np.mat(vector)
            return self.sizecheck(mat)
        if isinstance(vector,np.matrix): #只需要比较mat的大小
            return self.sizecheck(vector)
        if isinstance(vector,torch.Tensor):
            vector=np.mat(vector.detach().numpy())
            return self.sizecheck(vector)
        if isinstance(vector,int):
            vector=np.mat(vector)
            return self.sizecheck(vector)
        if isinstance(vector,np.int32):
            vector=np.mat(vector)
            return self.sizecheck(vector)
        if isinstance(vector,np.ndarray):
            vector=np.mat(vector)
            return self.sizecheck(vector)


        # observation, reward, done, info=self.pointee.stepin(action)
        # return observation, reward, done, info
    def sizecheck(self,mat):
        if np.size(mat,0)<np.size(mat,1):
            mat=mat.transpose()
            return mat
        else:
            return mat
    def Getobservation(self,state):
        state=self.typecheck(state)
        obs=self.pointee.pointee.calcobs(state)
        return obs
    def Discretesample(self):#提供随机功能，压缩到0~1之间，需要自己放大
        if(hasattr(self.pointee.pointee, 'sampleaction')):
            act=self.pointee.pointee.sampleaction()
            return act
        else:
            print("sample function not defined")
            return 0
    def Continuesample(self):#提供随机功能，压缩到0~1之间，需要自己放大
        if(hasattr(self.pointee.pointee, 'sampleaction')):
            act=self.pointee.pointee.sampleaction()
            return act
        else:
            print("sample function not defined")
            return 0