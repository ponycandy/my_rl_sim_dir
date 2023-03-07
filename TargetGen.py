import numpy as np
class TargetGen():
    def get_state(self,index):
        target_xy=np.mat("0;0")
        pos_oc=np.mat("0;0")
        vel_oc=np.mat("0;0")
        if(index==0):
            target_xy[0,0]=0
            target_xy[1,0]=2.309401077
            return target_xy,pos_oc,vel_oc
        if(index==1):
            target_xy[0,0]=-2
            target_xy[1,0]=-1.154700538
            return target_xy,pos_oc,vel_oc
        if(index==2):
            target_xy[0,0]=2
            target_xy[1,0]=-1.154700538
            return target_xy,pos_oc,vel_oc
#目标位置或者速度：
    #边长为4的等边三角
    #领导者在中心
    #最终速度为静止
    def evolve(self):
        #目标位置不发生改变
        pass
    def reset(self):
        pass