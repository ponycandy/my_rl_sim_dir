import json
import mysql.connector
import torch
from StringCacter import cactstring
mydb = mysql.connector.connect(
    host="127.0.0.1",       # 数据库主机地址
    user="root",    # 数据库用户名
    passwd='138797'  , # 数据库密码
    auth_plugin='mysql_native_password'
)
mycursor = mydb.cursor()
mycursor.execute("use "+"test_1")
# vector=torch.tensor([1,2,3])


# sql_input = "INSERT INTO runoob_tbl(runoob_id, runoob_title, runoob_author,submission_date) VALUES(%s, %s,%s, %s)"
# mycursor.execute(sql_input, [1, 2, 3,'2000-12-12'])

# l1 = [1,157.0,421.23]
# s = json.dumps(l1)
# sql_input = "INSERT INTO runoob_tbl(runoob_id, runoob_title, runoob_author,submission_date) VALUES(%s, %s,%s, %s)"
# mycursor.execute(sql_input, [2, s, 3,'2000-12-12'])

# sql_input = "INSERT INTO runoob_tbl(runoob_id, runoob_title, runoob_author,submission_date) VALUES(%s, %s,%s, %s)"
# mycursor.execute(sql_input, [3, s, 3,'2000-12-12'])

# l1 = [[1,157.0,421.23]]
# s = json.dumps(l1)
# sql_input = "INSERT INTO runoob_tbl(runoob_id, runoob_title, runoob_author,submission_date) VALUES(%s, %s,%s, %s)"
# mycursor.execute(sql_input, [4, s, 3,'2000-12-12'])

#清除表
# mycursor.execute("TRUNCATE TABLE runoob_tbl")


#创建表
# header1="varablename"
# header2="varablevalue"
# sql_input=cactstring("CREATE TABLE my_first_list ({} VARCHAR(255), {} VARCHAR(255) )",[header1,header2])
# mycursor.execute(sql_input)
#插入数据
# l1 = [[1,157.0,421.23]]
# s1 = json.dumps(l1)
# l2="test_get"
# sql_input = "INSERT INTO my_first_list(varablename, varablevalue) VALUES(%s, %s)"
# mycursor.execute(sql_input, [l2, s1])
# mydb.commit()


#提取数据
sql = "SELECT * FROM testtable WHERE COMMON_ID = %s"
na = [2]
mycursor.execute(sql, na)
myresult = mycursor.fetchone()
print(myresult)
# #使用数据，反向序列化
# l2 = json.loads(myresult[1])
# l1=myresult[0]
# print(l1)
# print(l2)
#存储tensor值可行！！
#清空数据表：


