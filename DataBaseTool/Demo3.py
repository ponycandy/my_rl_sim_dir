from Test_Complex_parser import complexparser
from DataBaseReader import Databasereader
import torch
import time
mdb=Databasereader()
mdb.USE_table("testtable")
# my_parser=complexparser()
# mdb.set_data_wrapper(my_parser)
#now run demo3 to test
#just use defauly parser is enough to handle tensor and RL issue
# for i in range(0,10000):
#     print(i)
#     my_tensor=torch.tensor([[1,2,3],[4,5,6]])
#     my_tensor1=torch.tensor([[10,20,30],[40,50,60]])
#     name="test_here"
#     mdb.Insert_data([my_tensor,my_tensor1,name]) #需要使用者自己保证头是对齐的，数目是正确的，这里不会自动检查


for i in range(0,10000):
    print(i)
    # my_tensor=torch.tensor([[1,2,3],[4,5,6]])
    # my_tensor1=torch.tensor([[10,20,30],[40,50,60]])
    # name="test_here"
    mdb.mycursor.execute("INSERT INTO testtable(header_1, header_2,header_3) VALUES(1, 2,3)")
mdb.mydb.commit()
#快了很多，但是相比还是很慢，可见我们的函数封包消耗了大量时间
#这对于提高RL的训练速度是绝无益处的
#放弃数据库方案，直接在现有的Replaybuffer上操作

# time_start = time.time()  # 记录开始时间
# mdb.mycursor.execute("SELECT * FROM  testtable WHERE COMMON_ID >= ((SELECT MAX(COMMON_ID) FROM testtable)-(SELECT  MIN(COMMON_ID) FROM testtable)) * RAND() + (SELECT MIN(COMMON_ID) FROM testtable)       LIMIT 1000")
# time_end = time.time()  # 记录结束时间
# time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
# print(time_sum)

#随机取样一千条的速度和replaybuffer是差不多的（当然，前提是不做任何直接转化，但是总体可以接受
# ，可是插入速度不可以接受，实在是太慢了）。我们还是需要把插入算法写在Replaybuffer里面，使用一些
#复杂算法将Replaybuffer和mysql连接起来，让数据可以存入文件
#但是，数据库的研究只能到目前为止了
# DAtalist=mdb.Get_data("COMMON_ID",4)
# print(DAtalist)
