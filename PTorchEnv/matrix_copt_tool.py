#说明，检视输入类型，并开辟内存，执行深度拷贝
import torch
from PTorchEnv.Typechecker import TensorTypecheck
def deepcopyMat(mat):
    mat=TensorTypecheck(mat)
    if isinstance(mat,torch.Tensor):
        return mat.detach().clone()
    if mat==None:
        return None
    else:
        return mat.copy()