from TCP_Manage import *
from Basic_env import Basic_env
class TCPenv(Basic_env):
    def __init__(self,port,IP):
        self.Matdatapointer,self.Matadatamanager=Tcpcreator.create_proxy(IP,port)
        self.set_pointee(self)
    def setstate(self,state):
        state=self.typecheck(state)
        self.Matadatamanager.sendMat(state)
        if(self.Matdatapointer.is_pudated()):##callback called
            callbackinfo=self.Matdatapointer.getdata()
            return callbackinfo
    def stepin(self,actionin):
        self.action=actionin
        self.Matadatamanager.sendMat(actionin)
        if(self.Matdatapointer.is_pudated()):
            self.statenext=self.Matdatapointer.getdata()
        if(hasattr(self.pointee, 'Info_extract')):
            info=self.pointee.Info_extract(self.statenext)
        else:
            info=0
        if(hasattr(self.pointee, 'missiondonejudge')):
            done=self.pointee.missiondonejudge(self.statenext)
        else:
            done=0
        return self.statenext,done,info
    def set_simer(self,pointee):
        self.pointee=pointee