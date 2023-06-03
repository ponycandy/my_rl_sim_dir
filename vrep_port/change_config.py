import json
import subprocess
with open("configdrone.json", "r", encoding='UTF-8') as f:
    params_dict = json.load(f)
port_prev=params_dict["port_num"]
cmd = "start_a_drone.bat "+str(port_prev)#同一文件夹下面
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)


with open("configdrone.json", "w", encoding='UTF-8') as f:
    json.dump(params_dict,f)