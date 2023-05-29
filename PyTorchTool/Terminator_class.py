class Terminator_class():
    def __init__(self,score_BASE_LINE,COUNT):
        self.score=score_BASE_LINE
        self.counter=COUNT
        self.steps=0
    def judge(self,scores):
        if(scores>self.score):
            self.steps+=1
        else:
            self.steps=0
        if(self.steps>self.counter):
            return 1
        else:
            return 0