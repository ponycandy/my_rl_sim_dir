import optuna
from DQN_Trial import DQN_Trial
from CartpoleTCP import CartpoleTCP
envnow=CartpoleTCP(8001,"127.0.0.1")
objtrial=DQN_Trial(4,3,[-100,0,100])
objtrial.set_env(envnow)
study = optuna.create_study()
func = lambda trial: objtrial.objective(trial,False, 200)
study.optimize(func, n_trials=100)


