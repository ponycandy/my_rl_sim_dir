import torch

from DeepMimic.SwarmTCP import SwarmTCP

envnow=SwarmTCP(8001,"127.0.0.1")

action=torch.Tensor([[1,2],[3,4],[5,6]])
while True:
    obs,reward,done,info= envnow.step(action)
    state=envnow.reset()
