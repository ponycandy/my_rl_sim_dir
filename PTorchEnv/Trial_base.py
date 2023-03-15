import optuna
import torch
from NNFactory import NNFactory
import numpy as np
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
            force_linear_model: bool = False,
            n_episodes_to_train: int = 200,
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
        self.agent = self.netfactory.create_agent(args)
        # train loop
        if hasattr(self.pointee,"TrainingLoop"):
            self.pointee.TrainingLoop(n_episodes_to_train,self.agent,args)
        else:
            print("Pointee not setteled!")
            return  0


        self.agent.save_model() #我们需要在actor_proxy内部置入FileManager，老早以前就完成了

        # evaluate its performance
        rewards, steps = self.evaluate(self.agent, self.envnow, n_episodes=1000, epsilon=0.00)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        return mean_reward


    # def TrainingLoop(self,n_episodes_to_train):
    #     #这里需要考虑解耦，允许使用不同的TrainingLoop直接定义
    #     #允许在DQN和DDPG以及其它RL算法之间作切换
    #     pass
    # def sample_hyper_parameters(self,trial):
    #     #这是一个自定义函数，仅在此说明其输入和输出类型
    #     #详细的定义在各衍生子类中完成
    #
    #     # 程序的运行逻辑：每次都在objective函数内部进行采样，如下面的例子：
    # # Categorical parameter
    # # optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])
    # #上面这个是说优化器在MomentumSGD和Adam之间可选
    # # # Integer parameter
    # # num_layers = trial.suggest_int("num_layers", 1, 3)
    # #上面这个是说，神经网络层数可选，使用工厂模式生成神经网络就可以满足神经网络超参可选了
    # # # Integer parameter (log)
    # # num_channels = trial.suggest_int("num_channels", 32, 512, log=True)
    # #神经网络的神经元个数可选
    # # # Integer parameter (discretized)
    # # num_units = trial.suggest_int("num_units", 10, 100, step=5)
    # #这个不知道
    # # # Floating point parameter
    # # dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)
    # #dropout率，不需要知道
    # # # Floating point parameter (log)
    # # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # #学习率可变
    # # # Floating point parameter (discretized)
    # # drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)
    #     #在上面suggest完成后，选择结果会直接在optuna里面进行贝叶斯排序，然后选择下一个合理的参数组，完成自动化训练
    #     #主程序的任务是将这些arg作为输入，绑定到DQN里面，需要使用工厂模式创建网络
    #     #如何绑定，见例子：https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
    #     #考虑到返回值是一系列数据,使用字典形式去输出
    #     # return args_dict
    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    def evaluate(self,agent, env, n_episodes=1000, epsilon=0.00):


        return rewards,steps