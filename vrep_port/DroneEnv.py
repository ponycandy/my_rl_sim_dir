import numpy as np

from Pyenv import Pyenv
from  DroneVrep import DroneVrep
import collections
class DroneEnv(Pyenv):
    def __init__(self,portnum=23000):
        super(DroneEnv, self).__init__()
        self.set_simer(self)
        self.drone=DroneVrep(portnum)
        self.target_pos_x=0
        self.target_pos_y=0
        self.target_pos_z=1
        self.target_pos_yaw=0
        self.steps=0
        self.nowakingcount=0
        self.nowakingflag=0

        #总是使用statenow表示当前状态，请注意：
    # statenow同样被parent级调用，所以不要更改变量名称！！
    def getreward(self,state,action):
    # 采用lunarland的方法设计奖励函数
    # 终结条件：飞行器范围超出
    # 核心函数QR误差函数
    # 注意终止条件的数值设置不应当导致飞行器自杀
    # 此外，还应该在理想状态下（到达目标位置的静态误差小于某一个值）的时候即及时终止
    # 也就是说，当飞行器位置和目标状态的范数小于一定值而且范数为小于一定值的时候，终止并给与额外奖励



        reward=self.statenow[2]*self.statenow[2]+self.statenow[3]*self.statenow[3]+self.statenow[4]*self.statenow[4]
        +self.statenow[5]*self.statenow[5]
        reward=-reward #误差奖励，QR的最优值
        if(abs(self.statenow[3])>5 or abs(self.statenow[4])>5 or abs(self.statenow[5])>5):
        #     这个时候已经超界了,强制结束并给与惩罚
            reward=reward-200
        else:
            if(abs(self.statenow[3])<0.1 and abs(self.statenow[4])<0.1 and abs(self.statenow[5])<0.1 and abs(self.statenow[6])<0.1
                    and abs(self.statenow[7])<0.1  and abs(self.statenow[8])<0.1):
                reward+=100
        #     完成任务并给与奖励
        if(self.nowakingflag==1):
            reward=reward-200
            self.nowakingflag=0
        return reward
    def calcobs(self,statevector):

        return  self.statenow
    def step_in(self,actionin):
        # 行动为1*4矩阵
        self.steps+=1
        vel_list=[actionin[0,0].item(),actionin[1,0].item(),actionin[2,0].item(),actionin[3,0].item()]
        self.drone.set_speed_list(vel_list)
        linear,angular=self.drone.get_drone_vel()
        pos=self.drone.get_drone_pos()
        orien=self.drone.get_drone_orien()
        self.statenow=[orien[0],orien[1],orien[2]-self.target_pos_yaw,
                       pos[0]-self.target_pos_x,pos[1]-self.target_pos_y,pos[2]-self.target_pos_z,
                       linear[0],linear[1],linear[2],
                       angular[0],angular[1],angular[2]]
        # vrep需要返回特征目标位置...
        # 现阶段暂时使用内部生成的目标位置
        return self.statenow
    # 角度和角速度以及速度都不需要归化，因为他们是有限的
    # 但是距离需要归化
    # 在DQN中的方法是限制最小和最大值，这里可以做类似的事情：
    # 位置超出限制之后强制终止
    # 实际使用中采用相对位置输入即可
    # 这里的输入也采用相对位置输入作为特征工程
    def missiondonejudge(self,statenext):
        if(abs(self.statenow[3])>5 or abs(self.statenow[4])>5 or abs(self.statenow[5])>5):
            return 1  #超限
        if(abs(self.statenow[3])<0.1 and abs(self.statenow[4])<0.1 and abs(self.statenow[5])<0.1 and abs(self.statenow[6])<0.1
                and abs(self.statenow[7])<0.1  and abs(self.statenow[8])<0.1):
            return 1  #达标
        if(self.not_waking()):
            self.nowakingflag=1  #未离地
            return 1
        return 0
    def sampleaction(self):
        return np.mat(collections.random())
    def randonsample(self):
        # 这里是对环境的初始化，使用set方法
        zpos=2
        self.drone.set_pos([0,0,zpos])
        pos=[0,0,zpos]
        orien=[0,0,0]
        linear=[0,0,0]
        angular=[0,0,0]
        self.drone.set_orien([0,0,0])
        self.statenow=[orien[0],orien[1],orien[2]-self.target_pos_yaw,
                       pos[0]-self.target_pos_x,pos[1]-self.target_pos_y,pos[2]-self.target_pos_z,
                       linear[0],linear[1],linear[2],
                       angular[0],angular[1],angular[2]]
        return self.statenow
    def not_waking(self):
        # 只要落地就算输
        if(abs((self.statenow[5]+self.target_pos_z))<0.3):
            self.nowakingcount+=1
        else:
            self.nowakingcount=0
        if(self.nowakingcount>1000):
            done=1
        else:
            done=0
        return done
    def Info_extract(self,statevector):

        return 1
class DroneEnvpretrain1(Pyenv):
    def __init__(self,portnum=23000):
        super(DroneEnvpretrain1, self).__init__()
        self.set_simer(self)
        self.drone=DroneVrep(portnum)
        self.target_pos_x=0
        self.target_pos_y=0
        self.target_pos_z=1
        self.target_pos_yaw=0
        self.steps=0
        self.nowakingcount=0
        self.nowakingflag=0

        #总是使用statenow表示当前状态，请注意：
    # statenow同样被parent级调用，所以不要更改变量名称！！
    def getreward(self,state,action):
    # 采用lunarland的方法设计奖励函数
    # 终结条件：飞行器范围超出
    # 核心函数QR误差函数
    # 注意终止条件的数值设置不应当导致飞行器自杀
    # 此外，还应该在理想状态下（到达目标位置的静态误差小于某一个值）的时候即及时终止
    # 也就是说，当飞行器位置和目标状态的范数小于一定值而且范数为小于一定值的时候，终止并给与额外奖励



        reward=self.statenow[2]*self.statenow[2]+self.statenow[3]*self.statenow[3]+self.statenow[4]*self.statenow[4]
        +self.statenow[5]*self.statenow[5]
        reward=-reward #误差奖励，QR的最优值
        if(abs(self.statenow[3])>5 or abs(self.statenow[4])>5 or abs(self.statenow[5])>5):
        #     这个时候已经超界了,强制结束并给与惩罚
            reward=reward-200
        else:
            if(abs(self.statenow[3])<0.1 and abs(self.statenow[4])<0.1 and abs(self.statenow[5])<0.1 and abs(self.statenow[6])<0.1
                    and abs(self.statenow[7])<0.1  and abs(self.statenow[8])<0.1):
                reward+=100
        #     完成任务并给与奖励
        if(self.nowakingflag==1):
            reward=reward-200
            self.nowakingflag=0
        return reward
    def calcobs(self,statevector):

        return  self.statenow
    def step_in(self,actionin):
        # 行动为1*4矩阵
        self.steps+=1
        vel_list=[actionin[0,0].item(),actionin[1,0].item(),actionin[2,0].item(),actionin[3,0].item()]
        self.drone.set_speed_list(vel_list)
        linear,angular=self.drone.get_drone_vel()
        pos=self.drone.get_drone_pos()
        orien=self.drone.get_drone_orien()
        self.statenow=[orien[0],orien[1],orien[2]-self.target_pos_yaw,
                       pos[0]-self.target_pos_x,pos[1]-self.target_pos_y,pos[2]-self.target_pos_z,
                       linear[0],linear[1],linear[2],
                       angular[0],angular[1],angular[2]]
        # vrep需要返回特征目标位置...
        # 现阶段暂时使用内部生成的目标位置
        return self.statenow
    # 角度和角速度以及速度都不需要归化，因为他们是有限的
    # 但是距离需要归化
    # 在DQN中的方法是限制最小和最大值，这里可以做类似的事情：
    # 位置超出限制之后强制终止
    # 实际使用中采用相对位置输入即可
    # 这里的输入也采用相对位置输入作为特征工程
    def missiondonejudge(self,statenext):
        if(abs(self.statenow[3])>5 or abs(self.statenow[4])>5 or abs(self.statenow[5])>5):
            return 1  #超限
        if(abs(self.statenow[3])<0.1 and abs(self.statenow[4])<0.1 and abs(self.statenow[5])<0.1 and abs(self.statenow[6])<0.1
                and abs(self.statenow[7])<0.1  and abs(self.statenow[8])<0.1):
            return 1  #达标
        if(self.not_waking()):
            self.nowakingflag=1  #未离地
            return 1
        return 0
    def sampleaction(self):
        return np.mat(collections.random())
    def randonsample(self):
        # 这里是对环境的初始化，使用set方法
        state=[0 ,0 ,0 ,0 ,0 ,2 ,0 ,0 ,0,0,0,0]
        self.drone.set_pos([0,0,2])
        pos=[0,0,2]
        orien=[0,0,0]
        linear=[0,0,0]
        angular=[0,0,0]
        self.drone.set_orien([0,0,0])
        self.statenow=[orien[0],orien[1],orien[2]-self.target_pos_yaw,
                       pos[0]-self.target_pos_x,pos[1]-self.target_pos_y,pos[2]-self.target_pos_z,
                       linear[0],linear[1],linear[2],
                       angular[0],angular[1],angular[2]]
        return self.statenow
    def not_waking(self):
        # 如果一直贴在地上，就会导致这一条
        if(abs((self.statenow[5]+self.target_pos_z))<0.3):
            self.nowakingcount+=1
        else:
            self.nowakingcount=0
        if(self.nowakingcount>100):
            done=1
        else:
            done=0
        return done
    def Info_extract(self,statevector):

        return 1