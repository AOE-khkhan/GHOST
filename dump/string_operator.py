from brain_functions import brain_functions
from functions import *

class string_operator:
    def operateString(self,que,ans,infl,related,data):
        related = self.getRelated(data)
        
        print("related = {}".format(related))
        print("possible ans = {}".format(self.getAns(data)))
        
        #this maps the related que to data
        dataMap = [self.mapStrings(data,que[x]) for x in range(len(que))]
##        print(dataMap)
        
        #this gets the common que in ans   
        commonFormatQue = [self.getCommonFormat(que[x],ans[x]) for x in range(len(que))]
##        print(commonFormatQue)
        
        #this gets the common anns in que 
        commonFormatAns = [self.getCommonFormat(ans[x],que[x]) for x in range(len(ans))]
##        print(commonFormatAns)
        
        #this uses the datamap to turn the individual ans to look as that of data
        outputFormat = [self.formatOutput(dataMap[x],commonFormatAns[x]) for x in range(len(commonFormatAns))]
##        print(outputFormat)
        

        #this gets the maximum output format
        d = {}
        for n in range(len(outputFormat)):
            if outputFormat[n] in d.keys():
                d[outputFormat[n]] += infl[n]
            else:
                d.setdefault(outputFormat[n],infl[n])
                
        ind = [x for x in d.values()]
        index = ind.index(max(ind))
        output_format = [x for x in d.keys()][index]
        print("output format = {}".format(output_format))

        ilist = []
        while output_format in outputFormat:
            i = outputFormat.index(output_format)
            ilist.append(i)
            outputFormat[i] = "`"
        
        #stream lining the values from the output format gotten
        que = [que[x] for x in ilist]
        print(que)
        
        ans = [ans[x] for x in ilist]
        print(ans)
        
        dataMap = [dataMap[x] for x in ilist]
##        print(dataMap)
        
        commonFormatQue = [commonFormatQue[x] for x in ilist]
        print(commonFormatQue)

        commonFormatAns = [commonFormatAns[x] for x in ilist]
        print(commonFormatAns)
        
        #this is to get the value of relationship of the data values from relate and frequency
        self.calc_class(data)
        if len(commonFormatAns) > 1 and all([sorted(x[-1])==sorted(commonFormatAns[0][-1]) for x in commonFormatAns]):
            ansvalue = output_format.split()
            while "[var]" in output_format.split() and len(commonFormatAns[0][-1]) > 0:
                ansvalue[ansvalue.index("[var]")] = commonFormatAns[0][-1][0]
                commonFormatAns[0][-1].pop(0)
            ansvalue = " ".join(ansvalue).strip()
            print("ans = {}".format(ansvalue))
        else:
            ansRel1 = []
            checker = []
            for xv in range(len(commonFormatAns)):
                checker.append([])
                for x in commonFormatAns[xv][-1]:
                    val = []
                    valinfl = []
                    for xx in self.memory["data"][self.memory["input"] > 0]:
                        vinfl = self.getRelation(xx,x," ")
                        if vinfl > 0 and vinfl < 1:
                            checker[-1].extend(xx.split())
                            if self.getCommonFormat(x,que[xv]) != "[var]":
                                checker[-1].append("[in-que]")
                            if self.getCommonFormat(xx,que[xv])[0] != "[var]":
                                valinfl.append(vinfl)
                                val.append(xx)
        
                    if len(val) > 0:
                        hv = max(valinfl)
                        while hv != 0 and hv in valinfl:
                            value = val[valinfl.index(hv)]
                            if value not in ansRel1: 
                                ansRel1.append(self.formatOutput(dataMap[xv],self.getCommonFormat(value,que[xv])))
                                #print(commonFormatAns[xv],"--------",value,"-------",ansRel1[-1])
                            valinfl[valinfl.index(hv)] = 0
            ansRel2 = [x for x in checker]
            l = len(checker)
            checker = [b for a in checker for b in a]
            checker = [x for x in set(checker) if checker.count(x) > l//2]
        
            #words present in answers
            checker_strict = []
            for x in ans:
                checker_strict.extend(x.split())
            
            #getting the presensce of words in ans
            checker_sinfl = []
            checker_s = []
            for x in set(checker_strict):
                if checker_strict.count(x) > len(ans)//2:
                    checker_s.append(x)
                    checker_sinfl.append(formatVal(checker_strict.count(x)/len(checker_strict)))
            #words present all through the answers
            checker_strict = [x for x in set(checker_strict) if checker_strict.count(x) == len(ans)]
            
            ansRel = []
        
            inque = False
            if len(checker) > 0:
                if "[in-que]" in checker:
                    ansRel.extend(ansRel1)
                    inque = True
                    checker.remove("[in-que]")
                for xx in ansRel2:
                    for x in xx:
                        if all([v in x for v in checker_strict]):
                            if x not in ansRel: ansRel.append(x)
            m = 0
            anskeyli = [x for x in set(ansRel) if x.count("[var]") == output_format.count("[var]")]
            if output_format in anskeyli:
                anskeyli = [output_format]
            value = []
            print(anskeyli)
            if len(anskeyli) > 0:
                val = []
                valinfl = []
                ansval = []        
                for anskey in anskeyli:
                    if "[var]" in anskey:
                        x = anskey#.replace("[var]","")
                        for xx in self.memory["data"][self.memory["input"] > 0]:
                            vinfl = self.getRelation(xx,x," ")
                            if vinfl > 0 and vinfl < 1:
                                if inque == True and self.getCommonFormat(xx,data)[0] != "[var]" and all([v in xx for v in checker_strict]) and all([ax in " "+xx+" " for ax in (" "+x.replace("[var]","").strip()+" ").split()]) and self.getRelation(xx,data) < 1:
                                    if xx not in val:
                                        for xy in set(xx.split()):
                                            if xy in checker_s:
                                                vinfl += checker_sinfl[checker_s.index(xy)]
                                        valinfl.append(vinfl)
                                        val.append(xx)
                                        r  = " ".join([x[0] for x in self.mapStrings(xx,anskey) if x[-1] == "[var]"])
                                        ansval.append(r)
                                        #print(r)
                print("val and valinfl")
                print("val = {}".format(val))
                print("valinfl = {}".format(valinfl))
                print("ansval = {}".format(ansval))
##                value = []
##                ansvalue = []
##                if len(val) > 0:
##                    hv = max(valinfl)
##                    while hv != 0 and hv in valinfl:
##                        value.append(val[valinfl.index(hv)])
##                        ansvalue.append(ansval[valinfl.index(hv)])
##                        valinfl[valinfl.index(hv)] = 0
                value = val
                ansvalue = ansval
                print(inque)
                print("checker = {}".format(checker_strict))
                print("anskeyli = {}".format(anskeyli))
                print("ansval = {}".format(ansval))
                print("value = {}".format(value))
                print("ansvalue = {}".format(ansvalue))
                ansvalue = [output_format.replace("[var]",x) for x in ansvalue]
                ansvalue = [x for x in ansvalue if len(x.strip()) > 0]
                for bx in self.getAns(data):
                    if bx not in ansvalue: ansvalue.append(bx)
                print("ans = {}".format(ansvalue))

                relate = False
                if type(ansvalue) == list:
                    if len(ansvalue) > 0: relate = True
                if type(ansvalue) == str:
                    if len(ansvalue.strip()) > 0: relate = True

                if relate:
                    varz = [x[-1] for x in commonFormatAns]

                    standard = self.class_filter(data,varz)
                    print("standard filter = {}".format(standard))
                    for x in ansvalue:
                        print(x,self.class_filter(data,[[x]]))

                relate = False    
                if type(ansvalue) == list:
                    if len(ansvalue) > 0: relate = True
                if type(ansvalue) == str:
                    if len(ansvalue.strip()) > 0: relate = True
                else: relate = False
                
                if relate:
                    reply = ansvalue
                    for datain in ["~",reply]:
                        if datain not in ["~","/"]:self.save2memory(datain)
            
                        self.setContext(datain)
                        
                        if datain not in ["~","/"]: self.classify(datain)
            
                        try:
                            if self.context[-2] == "~":
                                #self.mouth['speak'][self.context[-3]] = 1
                                self.addData(self.context[-3],datain, 1.0,"after")
                            else:
                                self.addData(self.context[-2],datain, 1.0,"after")
                                
                        except Exception as e:
                            print(e)
                        
                        if datain == "~":
                            self.switchSource()
                            self.manageSource()
                        
                        if datain not in ["~","/"]:
                            return self.compute(datain)
