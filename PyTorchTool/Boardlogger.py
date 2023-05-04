import torch
from tensorboardX import SummaryWriter
from datetime import datetime
class Boardlogger():
    def __init__(self):
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.writer =SummaryWriter("./my_log_dir/"+TIMESTAMP)
        self.step=0
        self.log_average_step=0
        self.log_average_step_internal=0
        self.log_average_value=0
    def log_per_step_scalr(self,value,name):
        self.writer.add_scalar(name,value,self.step)
        self.step+=1
    def log_average_per_step_scalr(self,value,name,steps):
        self.log_average_value+=value
        if self.log_average_step>0 and self.log_average_step%steps==0:
            self.writer.add_scalar(name,self.log_average_value/self.log_average_step,self.log_average_step_internal)
            self.log_average_value=0
            self.log_average_step=0
            self.log_average_step_internal+=1
        self.log_average_step+=1