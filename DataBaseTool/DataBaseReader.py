#声明：图像，视频以及过于巨大的原始训练数据，是不能够用DAtabase来储存的，只能用database来管理其地址
#varchar的最大长度位255字节，因此只能储存一些较小的张量，刚好足以应对我们的RL需求
import mysql.connector
from Tensorparser import Tensorparser
from StringCacter import IterateString_cut_END,SequenceReplace,IterateString_listcating
from DefaultParser import DefaultMaker
import json
import torch
class Databasereader():
    def __init__(self,Databasename="test_1",hostID="127.0.0.1",passwdcurrent='138797',username="root"):
        self.mydb = mysql.connector.connect(
            host=hostID,       # 数据库主机地址
            user=username,    # 数据库用户名
            passwd=passwdcurrent  , # 数据库密码
            auth_plugin='mysql_native_password'
        )
        self.initparser()
        self.mycursor = self.mydb.cursor()
        self.using_table_name=""
        if(self.Database_exist(Databasename)==1):
            self.mycursor.execute("use "+Databasename) #若采用默认输入，则会使用空表，否则总是会先创建新的表
        else:
            self.createDataBase(Databasename)
            self.use_Data_base(Databasename)
        pass
    def initparser(self):
        # self.parser_num=0
        self.parserdict=[]
        tensorparse=Tensorparser()
        self.parserdict.append(tensorparse)
        self.parser_num=len(self.parserdict)
        self.datamaker=DefaultMaker()


    def create_table(self,tablename,headerstr_list,datatype_list=[]):
        self.using_chart_name=tablename
        command=self.get_create_table_command(tablename,headerstr_list,datatype_list)
        self.mycursor.execute(command)
        command_1="ALTER TABLE "+ tablename +" ADD COLUMN COMMON_ID INT AUTO_INCREMENT PRIMARY KEY"
        self.mycursor.execute(command_1)
        print("chart has been created")

    def get_create_table_command(self,tablename,headerstr_list,datatype_list=[]):
        header_num=len(headerstr_list)
        command_1="CREATE TABLE "+tablename+"("+IterateString_cut_END(" HEADER_NAME HEADER_TYPE,",header_num)+")"
        if len(datatype_list)==0:
            datatype_list=["VARCHAR(255)" for _ in range(header_num)]
        else:
            pass
        command=SequenceReplace(command_1,"HEADER_NAME",headerstr_list)
        command=SequenceReplace(command,"HEADER_TYPE",datatype_list)
        #always give them a IP
        return command
    def USE_table(self,tablename):
        if self.Table_exist(tablename)==0:
            print("Table not exist! create one first!")
            return
        self.headernamelist=[]
        self.using_table_name=tablename
        command="show columns from "+self.using_table_name
        self.mycursor.execute(command)
        for x in self.mycursor:
            if x[0]=='COMMON_ID':
                pass
            else:
                self.headernamelist.append(x[0])
        #我们希望自动更新主键，所以需要获取表头名称，这一步在USE_TABLE里面完成
        data_num=len(self.headernamelist)
        self.insertDataTuple_namepart4= "  VALUES( "+IterateString_cut_END("%s,",data_num)+" ) "
        self.insertDataTuple_namepart3= " ( "+IterateString_listcating(self.headernamelist)+" ) "
        self.insertDataTuple_command="INSERT INTO "+ self.using_table_name+self.insertDataTuple_namepart3+self.insertDataTuple_namepart4

    def add_parser(self,dataparser):
        self.parserdict.append(dataparser)
        self.parser_num=len(self.parserdict)
    def deserilize(self,data_all):
        targetdata=self.datamaker.make(data_all)
        return targetdata

        #呃，这部分就不行了，依赖于具体的解释器
        #即使是同样的list表观也可能表达不同的意思
        #所以这里只进行最简单的反序列化，剩下的解释权需要parser去做

        #设计的默认parser会把所有的list解释为tensor
        #所以还是要一个外部的解释器！！！
        pass
    def serilize(self,list_member):
#不论输入什么都必须序列化！！
        if isinstance(list_member,float) or isinstance(list_member,int) or isinstance(list_member,str):
            return json.dumps(list_member)
        if list_member==None:
            return json.dumps("None")
        if isinstance(list_member,list):
            singlelist_serilized = json.dumps(list_member)
            return singlelist_serilized
        if isinstance(list_member,torch.Tensor):
            singlelist_serilized = json.dumps(list_member.tolist())
            return singlelist_serilized
        #above is the basic type that can be config autonomouslly
        #below is the custom Type iterator
        for i in range(0,self.parser_num):
            parser=self.parserdict[i]
            data=parser.parse(list_member)
            if data!=None:
                return data
    def Insert_data(self,datatuple):
        targetlist=[]
        length=len(datatuple)
        for i in range(0,length):
            singlelist=datatuple[i]
            singlelist_transformed=self.serilize(singlelist)
            targetlist.append(singlelist_transformed)
        self.insertDataTuple(targetlist)
    def Get_data(self,table_header,equal_value):
        datatuple=self.get_data_single_Tuple(table_header,equal_value)
        targetlist=self.deserilize(datatuple)
        return targetlist
    def Get_data_random_Batch(self,BatchSize):

        pass#return list of multiple document,each document is a list
    def get_one_by_property(self,condition,property):
        #根据表格的单个属性，如ID选取属性，需要用户保证该属性与数据是一对一的关系
        pass
    def insertDataTuple(self,data_Tuple):
        #此函数用来插入基础的string,float数据，但是不负责插入list复合数据
        #所以，应该不需要mission_name,mission_name应该位于更上层的位置

        self.mycursor.execute(self.insertDataTuple_command,data_Tuple)
        self.mydb.commit()
    def insert_data_RL(self,data_Tuple):
        pass
    def get_data_single_Tuple(self,table_header,equal_value):
        #返回满足检索条件的第一个值
        sql = "SELECT * FROM "+self.using_table_name+" WHERE "+table_header +"  = %s"
        na = [equal_value]
        self.mycursor.execute(sql, na)
        myresult = self.mycursor.fetchone()
        return myresult
    def clear_current_table(self):
        self.mycursor.execute("TRUNCATE TABLE "+self.using_table_name)
    def Table_exist(self,tablename):
        self.mycursor.execute("SHOW TABLES")
        for x in self.mycursor:
            if(x[0]==tablename):
                return True
            else:
                pass
        return False
    def Database_exist(self,databasename):
        self.mycursor.execute("SHOW DATABASES")
        for x in self.mycursor:
            if(x[0]==databasename):
                return True
            else:
                pass
        return False
    def createDataBase(self,databasename):
        self.mycursor.execute("CREATE DATABASE "+databasename)
    def use_Data_base(self,database):
        self.mycursor.execute("use "+database)
    def createTableblank(self,tablename):
        self.mycursor.execute("CREATE TABLE "+tablename)
        #建立的默认表格结构是无论如何都无法使用的，所以，我们需要将表头作为输入