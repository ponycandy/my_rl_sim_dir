import math
import torch
class variance():
    def __init__(self):
        self.var_n=0
        self.var_S1=0
        self.var_S2=0
    def calc_Variance_Iterative(self,value_in):
        self.var_n+=1
        self.var_S1+=value_in
        self.var_S2+=value_in**2
    def get_variance(self):
        self.var_value=torch.sqrt(self.var_S2/self.var_n-(self.var_S1/self.var_n)**2)
        return self.var_value
    def Clear_out_varriance(self):
        self.var_n=0
        self.var_S1=0
        self.var_S2=0
class RL_Calculator():
    def __init__(self):
        self.var_tar_minus_net=variance()
        self.var_tar=variance()
    def calc_KV(self):
        pass
    def calc_Entropy(self):
        pass
    def calc_Residual_Varriance_Iterative(self,target_values,network_values):
        self.var_tar_minus_net.calc_Variance_Iterative(target_values-network_values)
        self.var_tar.calc_Variance_Iterative(target_values)
        fenzi=self.var_tar_minus_net.get_variance()
        fenmu=self.var_tar.get_variance()
        return fenzi/fenmu
    def calc_Variance_Iterative(self,value_in,n_iterator,S1,S2):
        self.var_n+=1
        self.var_S1+=value_in
        self.var_S2+=value_in**2
        pass
    def get_Variance(self):
        self.var_value=torch.sqrt(self.var_S2/self.var_n-(self.var_S1/self.var_n)**2)
        return self.var_value
    def clear_out_varriance(self):
        self.var_n=0
        self.var_S1=0
        self.var_S2=0
        pass