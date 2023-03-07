
import numpy as np
from TCP_Manage import *
Matdatapointer,Matadatamanager=Tcpcreator.create_proxy('127.0.0.1',8001)
initial_state=np.mat("2,1,0,0,0; -2,-1,0,0,0;3,5,0,0,0")
Matadatamanager.sendMat(initial_state)