from zmqRemoteApi import RemoteAPIClient

class DroneVrep:
    def __init__(self):
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

        self.client.setStepping(True)
        self.sim.startSimulation()
        self.jointHandleList=[]
        index=0
        while index<4:
            handles=self.sim.getObject('./propeller['+str(index)+']/joint')
            self.jointHandleList.append(handles)
            index+=1
        self.body=self.sim.getObject('./base')
        self.drone=self.sim.getObject('./Quadcopter')

    def set_speed(self,index,velocity):
        if index==1 or index==3:
            velocity=-velocity
        self.sim.setJointTargetVelocity(self.jointHandleList[index],velocity)
        self.client.step()

    def set_speed_list(self,velocity_list):
        index=0
        velocity_list[1]=-velocity_list[1]
        velocity_list[3]=-velocity_list[3]
        while index<4:
            self.sim.setJointTargetVelocity(self.jointHandleList[index],velocity_list[index])
            index+=1
        self.client.step()

    def get_drone_orien(self):
        # simGetObjectOrientation will return the Euler
        # angles according to this convention. The values will be in radians, not degrees.
        # In CoppeliaSim, Tait-Bryan angles alpha, beta and gamma (or (a,b,g)) are used
        # 也就是自轴右旋转xyz轴顺序
        orien=self.sim.getObjectOrientation(self.drone,-1)
        return orien
    def get_drone_pos(self):
        pos=self.sim.getObjectPosition(self.drone,-1)
        return pos
    def get_drone_vel(self):
        linearVelocity,angularVelocity=self.sim.getObjectVelocity(self.drone)
        return linearVelocity,angularVelocity
    # 下面两个函数的bug在于：你不能够只设置base的位置，必须设置整个qudrocoptor的位置
    def set_pos(self,pos):
        self.sim.setObjectPosition(self.drone,-1,pos)
# sim.setObjectOrientation(drone,-1,[0.5,0.5,0.5])
    def set_orien(self,orien):
        self.sim.setObjectOrientation(self.drone,-1,orien)