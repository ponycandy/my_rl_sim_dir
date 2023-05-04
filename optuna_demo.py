import optuna
from DQN_Trial import DQN_Trial
from CartpoleTCP import CartpoleTCP
envnow=CartpoleTCP(8001,"127.0.0.1")
objtrial=DQN_Trial(4,2,[-15,15])
objtrial.set_env(envnow)
study = optuna.create_study()
func = lambda trial: objtrial.objective(trial,False, 1000)
study.optimize(func, n_trials=100)


