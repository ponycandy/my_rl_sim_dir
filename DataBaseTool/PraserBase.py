import json
class ParserBase():
    def __init__(self):
        pass
    def init_basic_parser(self,dict):
        self.tensorparse=dict["TensorPraser"]
        if(hasattr(self.pointee, 'extrainit')):
            self.pointee.extrainit(dict)
        else:
            pass
    def set_pointee(self,pointee):
        self.pointee=pointee
    def get_list_serilized(self,data):
        targetlist=[]
        if(hasattr(self.pointee, 'parselist')):
            list_of_all=self.pointee.parselist(data)
        else:
            # parser have no getlist function,
            # therefore data must all be simple list
            # or an error will be raised
            list_of_all=data
        datalen=len(list_of_all)
        for i in range(0,datalen):
            singlelist=list_of_all[i]
            if isinstance(singlelist,list):
                singlelist_serilized = json.dumps(singlelist)
            targetlist.append(singlelist_serilized)
        return targetlist
    def get_list_dserilized(self,data):
        targetlist=[]
        datalen=len(data)
        for i in range(0,datalen):
            singlelist=data[i]
            if isinstance(singlelist,list):
                singlelist_dserilized = json.loads(singlelist)
            targetlist.append(singlelist_dserilized)
        if(hasattr(self.pointee, 'makelist')):
            fetched_value=self.pointee.makelist(targetlist)
        else:
            fetched_value=targetlist

        return fetched_value
