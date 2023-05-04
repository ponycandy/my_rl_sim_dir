import optuna
import torch
from NNFactory import NNFactory
import numpy as np
from PTorchEnv.matrix_copt_tool import deepcopyMat
from PyTorchTool.FileManager import FileManager
class Trial_base():
    def __init__(self):
        self.Trial_Sequence=0
        self.netfactory=NNFactory()
        self.envnow=0
        self.pointee=0
        pass
    def set_pointee(self,_pointee):
        self.pointee=_pointee
    def set_env(self,env):
        self.envnow=env
    def objective(self,
            trial: optuna.trial.Trial,
            force_linear_model,
            n_episodes_to_train,
    ) -> float:
        #-> float  为python语法，检查返回值的类型
        #输入参数的前三个，是输入的类型检查
        """
        Samples hyperparameters, trains, and evaluates the RL agent.
        It outputs the average reward on 1,000 episodes.
        """

        # generate unique agent_id
        agent_id = self.Trial_Sequence
        self.Trial_Sequence+=1

        # hyper-parameters
        if hasattr(self.pointee,"sample_hyper_parameters"):
            args = self.pointee.sample_hyper_parameters(trial)
        else:
            print("Pointee not setteled!")
            return  0
        #args是RL的所有参数的集合，可以理解为list
        #为了方便以args['parameters']的形式调用其内部参数，往往将args定义为字典


        # 在参数重试中，必须保证随机种子是一定的，并且记录随机种子
        #但是实在不一致也没有banf
        # set_seed(env, args['seed'])

        # create agent object
        filsaved=FileManager()
        self.agent = self.netfactory.create_agent(args)
        # train loop
        if hasattr(self.pointee,"TrainingLoop"):
            self.pointee.TrainingLoop(n_episodes_to_train,self.agent,args)
        else:
            print("Pointee not setteled!")
            return  0


        # self.agent.save_model(agent_id) #我们需要在actor_proxy内部置入FileManager，老早以前就完成了
        torch.save(self.agent.actor, str(agent_id)+'.pt')
        # evaluate its performance
        n_episodes=5
        rewards, steps = self.evaluate(self.agent, n_episodes)
        mean_reward = -rewards/n_episodes
        return mean_reward
    def evaluate(self,agent, n_episodes):
        epoch=1
        lastobs=self.envnow.randominit()
        total_rewards=0
        stepdone=0
        agent.use_eps_flag=0
        while epoch<n_episodes:

            action=agent.response(lastobs)
            obs,reward,done,info=self.envnow.step(action)
            lastobs=deepcopyMat(obs)
            total_rewards+=reward
            stepdone+=1
            if done:
                epoch+=1
                lastobs=self.envnow.randominit()

        return total_rewards,stepdone