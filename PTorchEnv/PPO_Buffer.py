from PTorchEnv.Typechecker import TensorTypecheck
import torch
class PPO_Buffer():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    def appendnew(self,lastobs,action,obs,reward):
        # 类型检查是这里的职责
        lastobs=TensorTypecheck(lastobs)  #一般来说，lastobs输入默认为列向量
        self.states.append(lastobs.t())  #lastobs must be a*n tensor and this should be done outside here
        real_action=action[0]
        real_action_log=action[1]
        state_eval=action[2]
        self.actions.append(real_action)
        self.logprobs.append(real_action_log)
        self.state_values.append(state_eval)
        self.rewards.append(reward)
    def ResetNotify(self):
        self.is_terminals.append(True)
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
    def get_Batch_data(self):
        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.state_values, dim=0)).detach().to(self.device)
        return old_states,old_actions,old_logprobs,old_state_values