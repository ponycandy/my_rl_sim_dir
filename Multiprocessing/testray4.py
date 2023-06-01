import ray
import time
ray.init()
@ray.remote
class testRay():
    def __int__(self):
        pass
    def copyvalue(self,value):
        self.value=value
    def getvalue(self):
        return self.value
    def print_common_list(self,listco):
        self.value=listco[0]
        time.sleep(1)
polist=[]
for i in range(4):
    polist.append(testRay.remote())
i=0
for po in polist:
    po.copyvalue.remote(i)
    #所有ray远程对象的所有函数都只能使用.remote来调用，不然会报错
    #因为本质上，本地只具备收发TCP的功能，实际的值在另一个process里面
    i+=1
    print(ray.get(po.getvalue.remote()))
    #直接访问看起来也是不行的,只能使用返回函数访问,并且嵌套ray.get来抱证串行
    #基于以上两点我们可以对我们的代码进行debug
common_list=[1,2,3,4]
for po in polist:
#测试一下共读一个对象可能的bug
    po.print_common_list.remote(common_list)
for po in polist:
    print(ray.get(po.getvalue.remote()))
#上面这个过程将会并行执行,目前来看，没有什么bug..
#即使有些函数没有return值，我们依然可以使用ray.get来实现串行返回
for po in polist:
    ray.get(po.print_common_list.remote(common_list))
    print("return once")
#上面这些例子想必已经足够了
#一种设计模式是：使用ray.get串行
#另一种是使用一个return_id_list，最后get这个list都可以实现赋值,实现并行
#当然，后者高效很多


