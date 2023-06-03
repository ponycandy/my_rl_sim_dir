import json
import subprocess
with open("configdrone.json", "r", encoding='UTF-8') as f:
    params_dict = json.load(f)
params_dict["currnet_sequence"]+=1
params_dict["port_num"] += 4
params_dict["env_cmd"]="DroneEnv("+str(params_dict["port_num"])+")"
if(params_dict["currnet_sequence"]>params_dict["worker_num"]):
    params_dict["currnet_sequence"]=1
    params_dict["port_num"] = 23005
    params_dict["env_cmd"]="DroneEnv(23005)"


with open("configdrone.json", "w", encoding='UTF-8') as f:
    json.dump(params_dict,f)