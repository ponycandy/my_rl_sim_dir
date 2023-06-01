import ray
#下一个问题，remote线程是拷贝原环境还是操作于原环境上（指针传值还是copy传值）：
class dosthing_here():
    def __int__(self):
        self.value=[1,2,3,4]
    def gl_init(self):
        self.value = [1, 2, 3, 4]
    def set_value(self,value,pos):
        self.value[pos]=value
        return self.value
@ray.remote
class operate():
    def __int__(self):
        pass
    def set_classer(self,dosomething):
        self.dothing=dosomething
        return 1
    def set_value(self,pos,value):
        return  self.dothing.set_value(value,pos)
    def get_dict_test(self,params):
        self.pars=params['a']
polist=[]
for i in range(4):
    polist.append(operate.remote())
objn=dosthing_here()
objn.gl_init()
results_id_list=[]
for po in polist:
    results_id_list.append(po.set_classer.remote(objn))
    #这里将同一个对象设置到每个remote中
ray.get(results_id_list)

results_id_list=[]
pos=0
value=10
for po in polist:
    results_id_list.append(po.set_value.remote(pos,value))
    pos+=1
ray.get(results_id_list)
print(results_id_list)
#我们可以得到返回的值，但是返回的值是一个objhandle类型，而不是一个和testray4中一样的简单值
#也就是说,如果返回值太过复杂就会出问题？
#这个问题是可以绕过去的，只要我们每次都重写PPO_instance类的环境初设就行......
#现在先不考虑这个接口设计问题吧


#下面测试一下读取dict是否可行
results_id_list=[]
tinydict = {'a': 1}
for po in polist:
    results_id_list.append(po.get_dict_test.remote(tinydict))

#暂时来说没有报错


