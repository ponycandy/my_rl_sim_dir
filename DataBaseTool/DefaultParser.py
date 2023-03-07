import json
import torch
class DefaultMaker():
    def __init__(self):
        pass
    def make(self,data):
        #所有数据都必须使用json反向序列化
        #因为基本上采取的默认储存方式都是VARCHAR
        #当使用非VARCHAR类型时，请自写datamaker对象
        targetlist=[]
        length=len(data)
        for i in range(0,length-1):
            real_data=json.loads(data[i])
            if isinstance(real_data,list):
                real_data=torch.tensor(real_data)
            targetlist.append(real_data)
        return targetlist
