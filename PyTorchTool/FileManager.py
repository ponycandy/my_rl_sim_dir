import torch
import threading
import datetime as dt
#输入的model类型，均为继承了torch.nn.Module的类
#使用方法：1.令目标model继承这个类 2.兴建这个类对象，然后调用set_auto_mode和save_model即可
#可以将这个类集成到actor_proxy里面
class FileManager():
    def init_file_manage(self):
        self.saving_flag=0
        self.saving_count=0
        self.file_number=0
        self.savingiteration=500
    def set_auto_mode(self,model_obj,name,interior_seconds):
        self.model_obj=model_obj
        self.name=name
        t = threading.Timer(interior_seconds, self.save_model())
        t.start()
        #最好不要使用这个模式，否则可能会在训练一半的时候直接保存，数据出错d
        pass
    def save_model(self):
        now_time = dt.datetime.now().strftime("%F-%H%M%S")
        torch.save(self.model_obj, self.name+now_time+'.pt')
        #name是一个字符串类型，比如:'network'，无需输入后缀名，这里会自动补全
    def save_model(self,model,name):
        now_time = dt.datetime.now().strftime("%F-%H%M%S")
        torch.save(model, name+now_time+'.pt')
    def set_counter_mode(self,model_obj,name,savingiteration):
        self.saving_count=0
        self.saving_flag=1
        self.model_obj=model_obj
        self.name=name
        self.savingiteration=savingiteration
        pass
    def counter(self):
        self.saving_count+=1
        if(self.saving_flag==1):
            if(self.saving_count>self.savingiteration):
                self.saving_count=0
                torch.save(self.model_obj, self.name+"midresults_count_"+str(self.file_number)+'.pt')
                self.file_number+=1
    # def load_model(self,name):
    #     #说明，总是不指定路径，存在工作目录中
    #     return torch.load(name)
    # def load_model_path(self,name,path):
    #     #指定路径加载，相对路径例子：
    #     #相对路径例子：../../datasets/lululu
    #     #和linux差不多
    #     pass
    # def save_as_onnx(self,model,name):#将网络保存为onnx形式
    #     self.save_model(model,name+'.pt')
    #     torch_model = torch.load(name+'.pt') # pytorch模型加载
    #     batch_size = 1  #批处理大小,暂时默认为1
    #     input_shape = model.get_input_shape()   #输入数据形状，考虑到后面可能会遗忘其意义
    #     torch_model.eval()
    #     x = torch.randn(batch_size,*input_shape)
    #     export_onnx_file = name+".onnx"
    #     torch.onnx.export(torch_model,
    #                       x,
    #                       export_onnx_file,
    #                       opset_version=10,
    #                       do_constant_folding=True,	# 是否执行常量折叠优化
    #                       input_names=["input"],		# 输入名
    #                       output_names=["output"],	# 输出名
    #                       dynamic_axes={"input":{0:"batch_size"},		# 批处理变量
    #                                     "output":{0:"batch_size"}})
    #     #建议这里自动化
    # def save_model_param(self,model,name):
    #     torch.save(model.state_dict(), name+'_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)
    # def load_params(self,model,name):
    #     model.load_state_dict(torch.load(name+'_params.pkl'))
    # def automated_save_param(self,option,model,savingiteration=100):
    #     self.mmodel=model
    #     if(option==1):
    #         self.saving_flag=1
    #     else:
    #         self.saving_flag=0
    #     self.savingiteration=savingiteration

    #     pass