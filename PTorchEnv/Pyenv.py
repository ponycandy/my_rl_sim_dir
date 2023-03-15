from Basic_env import Basic_env
class Pyenv(Basic_env):
    def __init__(self):
        super(Pyenv, self).__init__()
        self.set_pointee(self)
        self.statenow=0
        self.statedim=0
        self.actiondim=0
    def setstate(self,state):
        state=self.typecheck(state)
        self.statenow=state
        return 1
    def stepin(self,actionin):
        self.action=self.typecheck(actionin)
        self.statenext=self.typecheck(self.step_in(actionin))
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