import sim # import the sim.py file
import math # import the math module
sim.simxFinish(-1) # close any previous connection
clientID = sim.simxStart('127.0.0.1', 19991, True, True, 5000, 5) # connect to CoppeliaSim
if clientID != -1: # connection successful
    print('Connected to CoppeliaSim')
    # get the handle of the joint object
    returnCode, jointHandle = sim.simxGetObjectHandle(clientID, 'joint', sim.simx_opmode_blocking)
    if returnCode == sim.simx_return_ok: # handle retrieved successfully
        print('Got joint handle')
        # set the joint target position to 90 degrees
        targetPosition = math.pi / 2 # in radians
        returnCode = sim.simxSetJointTargetPosition(clientID, jointHandle, targetPosition, sim.simx_opmode_blocking)
        if returnCode == sim.simx_return_ok: # position set successfully
            print('Set joint target position')
        else:
            print('Failed to set joint target position')
        returnCode = sim.simxSetJointForce(clientID, jointHandle, 0.001, sim.simx_opmode_blocking)
        if returnCode == sim.simx_return_ok: # position set successfully
            print('Set joint target force')
        else:
            print('Failed to set joint target force')
    else:
        print('Failed to get joint handle')
    sim.simxFinish(clientID) # close the connection
else:
    print('Failed to connect to CoppeliaSim')
