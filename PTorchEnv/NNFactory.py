import torch
from DQN_Actor_Proxy import DQN_Actor_Proxy
from AC_Critic_Template import AC_Critic_Template
from DDPG_Actor_Proxy import DDPG_Actor_Proxy
from DDPG_critic_Proxy import DDPG_critic_Proxy
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
        if  args["Net_option"]=="DDPG_Net":
            return self._net_create_ddpg(args)


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
    def _net_create_ddpg(self,args):
        Net_list={}
        actornet,criticNet=self._critic_net_create(args)
        Net_list["actornet"]=actornet
        Net_list["criticNet"]=criticNet
        return Net_list#两个网络，一个是critic网络，一个是actor网络
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
        if  args["Net_option"]=="DDPG_Net":
            proxy_list={}
            Actor_proxy=DDPG_Actor_Proxy()
            Critic_Proxy=DDPG_critic_Proxy()
            Actor_proxy.setNet(actor_Net["actornet"].to(self.device))
            Critic_Proxy.setNet(actor_Net["criticNet"].to(self.device))
            proxy_list["actor"]=Actor_proxy
            proxy_list["critic"]=Critic_Proxy
            return proxy_list
    def _proxy_create_dqn(self,args,actor_Net):
        pass
    def _critic_net_create(self,args):
        action_channel=[]
        state_channel=[]
        output_channel=[]
        len_act=args["len_act"]
        len_state=args["len_state"]
        len_output=args["len_output"]
        n_units=[]
        for i in range(0,len_act):
            n_units.append( args["action_n_units_l{}".format(i)])
            action_channel=self._create_forward(args["outputnum_int"], args["Merge_results"],n_units)
        n_units=[]
        for i in range(0,len_state):
            n_units.append( args["state_n_units_l{}".format(i)])
            state_channel=self._create_forward(args["inputnum_int"], args["Merge_results"],n_units)
        n_units=[]
        for i in range(0,len_output):
            n_units.append( args["output_n_units_l{}".format(i)])
            output_channel=self._create_forward(args["Merge_results"],1,n_units)
        criticNet=AC_Critic_Template(action_channel,state_channel,output_channel)
        # criticNetdone
        num_layers=args["num_layers"]
        layers = []
        in_size=args["inputnum_int"]
        for i in range(num_layers):
            n_units = args["n_units_l{}".format(i)]
            layers.append(torch.nn.Linear(in_size, n_units))
            layers.append(torch.nn.ReLU())
            in_size = n_units
        layers.append(torch.nn.Linear(in_size, args["outputnum_int"]))
        layers.append(torch.nn.Tanh())
        actorNet=torch.nn.Sequential(*layers)

        return actorNet,criticNet
    def _create_forward(self,inputsize,outputsize,units_list):
        num_layers=len(units_list)
        layers = []
        in_size=inputsize
        for i in range(num_layers):
            n_units = units_list[i]
            layers.append(torch.nn.Linear(in_size, n_units))
            layers.append(torch.nn.ReLU())
            in_size = n_units
        layers.append(torch.nn.Linear(in_size, outputsize))
        return layers