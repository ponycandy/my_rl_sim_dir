from TCP_Manage import *
from Basic_env import Basic_env
from PTorchEnv.Typechecker import TensorTypecheck
class TCPenv(Basic_env):
    def __init__(self,port,IP):
        super(TCPenv, self).__init__() #首先调用父类的初始化函数进行初始化
        self.Matdatapointer,self.Matadatamanager=Tcpcreator.create_proxy(IP,port)
        self.set_pointee(self)
        if(hasattr(self.pointee.pointee, 'calcobs')==False):
            assert False,"calcobs function not define!"
        if(hasattr(self.pointee, 'Info_extract')==False):
            assert False,"Info_extract function not define!"
        if(hasattr(self.pointee, 'missiondonejudge')==False):
            assert False,"missiondonejudge function not define!"
    def setstate(self,state):

        obs=TensorTypecheck(self.pointee.pointee.calcobs(self.typecheck(state))).to(self.device)

        #可能的error,当obs和state不一致的时候，这里要出错.....
        state=self.typecheck(state)
        self.Matadatamanager.sendMat(state)
        if(self.Matdatapointer.is_pudated()):##callback called
            callbackinfo=self.Matdatapointer.getdata()
            # return callbackinfo
        return obs
    def stepin(self,actionin):
        self.action=actionin
        self.Matadatamanager.sendMat(actionin)
        if(self.Matdatapointer.is_pudated()):
            self.statenext=self.Matdatapointer.getdata()

        info=self.pointee.Info_extract(self.statenext)


        done=self.pointee.missiondonejudge(self.statenext)

        return self.statenext,done,info
    def set_simer(self,pointee):
        self.pointee=pointee