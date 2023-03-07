def cactstring(origin_string,inputlist):
    command=".format("
    iterate=len(inputlist)
    for i in range(0,iterate):
        command=command+"inputlist["+str(i)+"],"
    command="origin_string"+command+")"
    newstring=eval(command)
    #使用的例子：cactstring("testonce {} {}",["apple","bananas"])
    return newstring

def replace_string(origin_string,replacee,replacer):

    pass
def IterateString_listcating(string_list,comma=","):
    length=len(string_list)
    command=""
    for i in range(0,length):
        command=command+string_list[i]+comma
    return command.rstrip(comma)
def IterateString(stringTuple,times):
    command=""
    for i in range(0,times):
        command=command+stringTuple
    return command

def SequenceReplace(origin,replacee,replacer_list):
    origin=origin.replace(replacee, "{}")
    origin=cactstring(origin,replacer_list)
    return origin

def IterateString_cut_END(stringTuple,times,cutend=[]):
    if len(cutend)==0:
        cutend=stringTuple[-1]
    command=""
    for i in range(0,times):
        command=command+stringTuple
    return command.rstrip(cutend)

    # rstrip(',')