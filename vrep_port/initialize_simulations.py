import json
import subprocess
with open("configdrone.json", "r", encoding='UTF-8') as f:
    params_dict = json.load(f)
workernum=params_dict["worker_num"]
port_prev = params_dict["port_num"]
PID_list=[]

for i in range(workernum):
    cmd = "start_a_drone.bat " + str(port_prev)  # 同一文件夹下面
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    pid = p.pid
    dict_i={"pid":pid}
    PID_list.append(dict_i)
    port_prev+=4
nulldict={"process":PID_list}

with open("PIDS.json", "w", encoding='UTF-8') as f:
    json.dump(nulldict,f)