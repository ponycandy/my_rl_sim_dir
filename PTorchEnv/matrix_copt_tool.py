#说明，检视输入类型，并开辟内存，执行深度拷贝
import torch
def deepcopyMat(mat):
    if isinstance(mat,torch.Tensor):
        return mat.detach().clone()
    if mat==None:
        return None
    else:
        return mat.copy()