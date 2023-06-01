# 以下列写教程网站：
# https://medium.com/codable/ray-tutorial-b3a1fb6e3601
# https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8?gi=cd5b7eb3db70
# 经过思考，我发现如果每个进程只负责收集经验，而不负责update，我们将会再代码修改上面付出较大的代价：
# https://blog.csdn.net/weixin_43145941/article/details/116764574
# 使用第一个思路比较好
# 之所以说会付出较大的代价，是因为我们将会不得不重写优势函数计算的整个过程
# 这个过程是极度error_prone的，即使我们自认为前期的接口设计已经尽可能解耦了
#我们在此处的基本思想是，借助多线程来探索环境，然后在主线程中实现更新，然后再拿更新后的网络来更新环境
#伪代码如下：
# while True:
#     fill_in_buffer()
#     optimizer.update()
#     sync_net_2_all_proc()
#
# 先照着做？
#第一步，要将PPO变成一个训练类，我们继续展开上面的伪代码：
# for i=1:1:4
#    PPO_list[i]=PPO.remote()
#第二步，建立同步参数的函数：
# def sync_proc_net(PPO_list):
#     act_model_dict = net.act.state_dict()
#     cri_model_dict = net.cri.state_dict()
#     for k1,k2 in zip(act_model_dict.keys(),cri_model_dict.keys()):
#         result1 = torch.zeros_like(act_model_dict[k1])
#         result2 = torch.zeros_like(cri_model_dict[k2])
#         for j in range(process_num):
#             result1 += net_list[j][0].state_dict()[k1]
#             result2 += net_list[j][1].state_dict()[k2]
#         result1 /= process_num
#         result2 /= process_num
#         act_model_dict[k1] = result1
#         cri_model_dict[k2] = result2
#     net.act.load_state_dict(act_model_dict)
#     net.cri.load_state_dict(cri_model_dict)

# 第三步建立PPO的子列表
# while True:
#     for i=1:1:4
#        PPO_list[i]=PPO.remote()
#     all_process_epoch_done=ray.get(return_all_done_flag.remote)
#     average_state_dict=sync_state_dict
#便利所有PPO进程，同步神经网络的参数，这是所有进程唯一有交集的地方
#     for i=1:1:4
#        PPO_list[i].load_net=PPO.remote()
#使用ray的方式实现多线程，OK,I am exciting!