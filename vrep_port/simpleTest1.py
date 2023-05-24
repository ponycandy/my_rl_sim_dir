from  DroneVrep import DroneVrep


drone=DroneVrep()
testnum=1.275
drone.set_speed(0,1)
pos=drone.get_drone_pos()
orien=drone.get_drone_orien()
omega=drone.get_drone_omega()
linear,angular=drone.get_drone_vel()
print(linear)
