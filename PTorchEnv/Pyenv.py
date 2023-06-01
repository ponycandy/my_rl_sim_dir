from Basic_env import Basic_env
from PTorchEnv.Typechecker import TensorTypecheck
class Pyenv(Basic_env):
    def __init__(self):
        super(Pyenv, self).__init__()
        self.set_pointee(self)
        self.statenow=0
        self.statedim=0
        self.actiondim=0
    def setstate(self,state):
        obs=TensorTypecheck(self.pointee.pointee.calcobs(self.typecheck(state))).to(self.device)
        #可能的error,当obs和state不一致的时候，这里要出错.....
        state=self.typecheck(state)
        self.statenow=TensorTypecheck(state)

        return obs
    def stepin(self,actionin):
        self.action=self.typecheck(actionin)
        self.statenext=self.typecheck(self.step_in(actionin))
        info=self.pointee.Info_extract(self.statenext)
        done=self.pointee.missiondonejudge(self.statenext)
        return self.statenext,done,info
    def set_simer(self,pointee):
        self.pointee=pointee
        if(hasattr(self.pointee, 'missiondonejudge')==False):
            assert False,"missiondonejudge function not define!"
        if(hasattr(self.pointee, 'Info_extract')==False):
            assert False,"Info_extract function not define!"
        if(hasattr(self.pointee, 'calcobs')==False):
            assert False,"calcobs function not define!"