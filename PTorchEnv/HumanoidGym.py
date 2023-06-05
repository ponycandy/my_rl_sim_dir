from Pyenv import Pyenv
import gym
import pybullet_envs
import torch
from PTorchEnv.Typechecker import TensorTypecheck

class HumanoidGym(Pyenv):
    def __init__(self):
        super(HumanoidGym, self).__init__()
        self.set_simer(self)
        self.steps=0
        self.envnow = gym.make('HumanoidBulletEnv-v0')
        self.envnow.reset()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def step_in(self,actionin):
        obs, reward, done, info = self.envnow.step(actionin)
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

        state= self.envnow.reset()
        self.obs=state
        return  self.obs