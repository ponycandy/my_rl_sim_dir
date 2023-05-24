import sim # import the sim.py file
import math # import the math module

# This small example illustrates how to use the remote API
# synchronous mode. The synchronous mode needs to be
# pre-enabled on the server side. You would do this by
# starting the server (e.g. in a child script) with:
#
# simExtRemoteApiStart(19997,1300,false,true)
# 注意，只有19997端接口能够步进，并且可以直接运行python文件不需要运行vrep
class DroneVrep:
    def __init__(self):
        sim.simxFinish(-1) # close any previous connection
        self.clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5) # connect to CoppeliaSim
        if self.clientID != -1: # connection successful
            print('Connected to CoppeliaSim')
        else:
            assert False,"Failed to Connect"
        codes=sim.simxSynchronous(self.clientID,True)
        sim.simxStartSimulation(self.clientID,sim.simx_opmode_oneshot)
        if codes != -1: # connection successful
            print('Connected to sync')
        else:
            assert False,"Failed to sync"
        self.jointHandleList=[]
        index=0
        while index<4:
            returnCode,handles=sim.simxGetObjectHandle(self.clientID, './propeller['+str(index)+']/joint', sim.simx_opmode_blocking)
            self.jointHandleList.append(handles)
            if returnCode == sim.simx_return_ok: # handle retrieved successfully
                print('Got joint handle')
            else:
                assert False,"Failed to get joint"
            index+=1
        self.body=sim.simxGetObjectHandle(self.clientID,'./base',sim.simx_opmode_blocking )
        self.body=self.body[1]



    def set_speed(self,index,velocity):
        if index==1 or index==3:
            velocity=-velocity
        returnCode = sim.simxSetJointTargetVelocity(self.clientID, self.jointHandleList[index], velocity, sim.simx_opmode_blocking)
        if returnCode == sim.simx_return_ok: # position set successfully
            print('Set joint target vel')
        else:
            assert False,'Failed to set joint target vel'
        sim.simxSynchronousTrigger(self.clientID)

    def set_speed_list(self,velocity_list):
        index=0
        velocity_list[1]=-velocity_list[1]
        velocity_list[3]=-velocity_list[3]
        while index<4:
            returnCode = sim.simxSetJointTargetVelocity(self.clientID, self.jointHandleList[index], velocity_list[index], sim.simx_opmode_blocking)
            if returnCode == sim.simx_return_ok: # position set successfully
                print('Set joint target vel')
            else:
                assert False,'Failed to set joint target vel'
            index+=1
        sim.simxSynchronousTrigger(self.clientID)

    def get_drone_orien(self):
        # simGetObjectOrientation will return the Euler
        # angles according to this convention. The values will be in radians, not degrees.
        # In CoppeliaSim, Tait-Bryan angles alpha, beta and gamma (or (a,b,g)) are used
        # 也就是自轴右旋转xyz轴顺序
        orien=sim.simxGetObjectOrientation(self.clientID,self.body,-1,sim.simx_opmode_blocking)
        return orien[1]
    def get_drone_pos(self):
        pos=sim.simxGetObjectPosition(self.clientID,self.body,-1,sim.simx_opmode_blocking)
        return pos[1]
    def get_drone_omega(self):
        vel=sim.simxGetObjectVelocity(self.clientID,self.body,sim.simx_opmode_blocking)
        omega=vel[2]
        return omega
    def get_drone_linearvel(self):
        vel=sim.simxGetObjectVelocity(self.clientID,self.body,sim.simx_opmode_blocking)
        linear=vel[1]
        return linear
    def get_drone_vel(self):
        vel=sim.simxGetObjectVelocity(self.clientID,self.body,sim.simx_opmode_blocking)
        linear=vel[1]
        omega=vel[2]
        return linear,omega
