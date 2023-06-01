from Pyenv import Pyenv
import gym
import torch
from PTorchEnv.Typechecker import TensorTypecheck

class CartPoleGym(Pyenv):
    def __init__(self):
        super(CartPoleGym, self).__init__()
        # self.set_pointee(self)
        self.set_simer(self)
        self.steps=0
        self.envnow = gym.make("CartPole-v1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def step_in(self,actionin):
        obs, reward, done, info, _ = self.envnow.step(actionin.item())
        self.reward=reward
        self.done=done
        if info:
            self.info="truncated"
        else:
            self.info="no"
        self.obs=TensorTypecheck(obs)
        return self.obs
    def getreward(self,state,action):
        return self.reward
    def calcobs(self,statevector):

        return  self.obs
    def Info_extract(self,statevector):

        return self.info
    def missiondonejudge(self,statevector):
        return self.done
    def randonsample(self):

        state,info= self.envnow.reset()
        self.obs=state
        return  self.obs