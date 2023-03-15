import torch
from DQN_Actor_Proxy import DQN_Actor_Proxy
class NNFactory():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pass
    def create_agent(self,args):
        actor_Net=self.Net_create(args)
        actor_Proxy=self.Proxy_create(args,actor_Net)
        return actor_Proxy
    def Net_create(self,args):
        if  args["Net_option"]=="DQN_Net":
            return self._net_create_dqn(args)


    def _net_create_dqn(self,args):
        num_layers=args["num_layers"]
        layers = []
        in_size=args["inputnum_int"]
        for i in range(num_layers):
            n_units = args["n_units_l{}".format(i)]
            layers.append(torch.nn.Linear(in_size, n_units))
            layers.append(torch.nn.ReLU())
            in_size = n_units
        layers.append(torch.nn.Linear(in_size, args["outputnum_int"]))

        return torch.nn.Sequential(*layers)
    def Proxy_create(self,args,actor_Net):
        if  args["Net_option"]=="DQN_Net":
            proxy=DQN_Actor_Proxy()
            proxy.setNet(actor_Net.to(self.device))
            proxy.set_action_num(args["outputnum_int"])
            proxy.setoutputList(args["output_act_list"])
            proxy.use_eps_flag=1
            proxy.learning_rate=args["learning_rate"]
            proxy.EPS_DECAY=args["EPS_DECAY"]
            return proxy
    def _proxy_create_dqn(self,args,actor_Net):
        pass