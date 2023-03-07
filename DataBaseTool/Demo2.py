from DataBaseReader import Databasereader
mdb=Databasereader()
mdb.USE_table("testtable")
# mdb.create_table("testtable",["header_1","header_2","header_3"])
# mdb.insertDataTuple([1,2,"hello"]) #我们希望自动更新主键，所以需要获取表头名称
# mdb.Insert_data([1,2,[1,2]])
# resu=mdb.get_data_single_Tuple("header_3","hello")

mdb.clear_current_table()
# 也就是默认中间不能够更换其余数据库，如果要更换，必须使用USE_table语句，该语句会自动将表头信息加载出来
#createtable会默认调用USE_table语句