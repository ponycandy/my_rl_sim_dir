import json
for i in range(3):
    with open("configdrone.json", "r", encoding='UTF-8') as f:
        params_dict = json.load(f)
    if "extra_cmd_prev" in params_dict:
        for command in params_dict["extra_cmd_prev"]:
            exec(command["cmd"])
    exec(params_dict["env_import"])

    env = eval(params_dict["env_cmd"])
    if "extra_cmd_after" in params_dict:
        for command in params_dict["extra_cmd_after"]:
            exec(command["cmd"])
