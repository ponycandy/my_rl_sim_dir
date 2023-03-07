import torch
import numpy as np
def TensorTypecheck(vector):##始终使得宽大于长
    # torch.tensor(threads.critic_input_k)
    if isinstance(vector,list): #按照list类型进行转化
        mat=torch.tensor(np.mat(vector))
        return sizecheck(mat)
    if isinstance(vector,np.matrix): #只需要比较mat的大小
        mat=torch.tensor(vector)
        return sizecheck(mat)
    if isinstance(vector,torch.Tensor):
        return sizecheck(vector)
    if isinstance(vector,int):
        mat=torch.tensor([[vector]])
        return mat
    if isinstance(vector,np.float64):
        mat=torch.tensor([[vector]])
        return mat
    if isinstance(vector,np.ndarray):
        vector=torch.from_numpy(vector)
        return sizecheck(vector)

    # observation, reward, done, info=self.pointee.stepin(action)
    # return observation, reward, done, info
def sizecheck(mat):
    if len(mat.shape)==2:
        if mat.size(0)>mat.size(1):
            mat0=mat.t()
            return mat0
        else:
            return mat
    if len(mat.shape)==1:
        mat0=mat.unsqueeze(1)
        return mat0
def double_typecheck(value):
    if isinstance(value,torch.Tensor):
        if len(value.shape)==2:
            value_1=value[0,0]
            value_2=value_1.tolist()
            value_3=value_2[0][0]
            return value_3
        if len(value.shape)==1:
            value_1=value[0]
            value_2=value_1.tolist()
            value_3=value_2[0]
            return value_3
        if len(value.shape)==0:
            value_1=value
            value_2=value_1.tolist()
            value_3=value_2
            return value_3
    if isinstance(value,float):
        return value
    if isinstance(value,int):
        return value

