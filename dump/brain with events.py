import time

from collections import Counter

from subsetter import subsetter
from terminal import terminal
from brain_functions import brain_functions
from functions import *

class intelligence(subsetter, terminal, brain_functions):
    def manageSource(self):
        if self.lastSource != self.source:
            self.source = self.lastSource

        self.save()
        
    def analyse(self, data):
        if data.startswith("?"):
            return self.runCommand(data)
        else:
            if data not in ["~","/"]:
                self.save2memory(data)
                
            self.setContext(data)

            try:
                if self.context[-2] == "~":
                    self.setReply(self.context[-3], data)
                
            except Exception as e:
                print(e)
                
        if data == "~":
            self.switchSource()
            self.manageSource()
            
        if data not in ["~","/"]:
            for x in data.split(" "):
                self.save2memory(x, 1)
##                self.generateArtificialData(x)
            if not self.learning:
                r = self.event_process(data)
                if r == None:
                    return self.compute(data)
                else:
                    return r
        
    def compute(self,data):
        self.show_process("data = {}".format(data))
        
        rel = self.getRelated(data, " ", 0.1, strict=True)
        related = rel.copy()             
        for x in rel:
            if len(x.split()) != len(data.split()):
                related[x] *= 0.5
        pd = self.sort_dict(related)
        print(related)
        print(pd)
        m = 0
        if pd[0][0] == data and len(pd) > 1: m = 1
        print(pd[m][-1]*0.5)
       
        related = {x:related[x] for x in related if related[x] >= pd[m][-1]*0.5}
        
        self.show_process("related = {}".format(related))
        relatedlist = [x for x in related]
        que, ans = self.getQueAns(relatedlist)

        ##------------------------this is for inheritance of properties when no ans---------------------------------------------------------------------
        rel = relatedlist.copy()
        rel.remove(data)
        
        #this maps the related que to data
        dataMap = [self.mapStrings(data, rel[x]) for x in range(len(rel))]

        #this uses the datamap to turn the individual que to look as that of data
        searchFormats = [self.getCommonFormat(rel[x],data) for x in range(len(rel))]
        searchFormat_values = [x[0] for x in searchFormats]
        
        for i in range(len(rel)): self.show_process("\nrel = {}\ndatamap = {}\nsearchFormat = {}\n".format(rel[i], dataMap[i], searchFormats[i]))

        searchFormat_classes = {data.replace(x.replace("[var]", "").strip(), "").strip():[i for i in range(len(searchFormat_values)) if x == searchFormat_values[i]] for x in set(searchFormat_values)}

        while "" in searchFormat_classes:
            searchFormat_classes.pop("")
            
        new_searchFormat_classes = {}
        for cl in searchFormat_classes:
            temp_searchFormats = [searchFormats[i] for i in searchFormat_classes[cl]]
            for x in range(len(rel)):
                if rel[x] in que:
                    if cl in new_searchFormat_classes:
                        new_searchFormat_classes[cl].append(x)
                    else:
                        new_searchFormat_classes.setdefault(cl, [x])
                            
        self.show_process("{} class(es) found: {}".format(len(searchFormat_classes), [x+" "+str(len(x)) for x in searchFormat_classes]))
        self.show_process("and {} more class(es) found: {}".format(len(new_searchFormat_classes), [x+" "+str(len(x)) for x in new_searchFormat_classes]))

        for cls in [searchFormat_classes, new_searchFormat_classes]:
            for cl in cls:
                temp_searchFormats = [searchFormats[i] for i in cls[cl]]
                
                #this inherits properties from the related values to generate more artificial data 
                if len(temp_searchFormats) > 1 and all([temp_searchFormats[0][0] in a for a in temp_searchFormats]):
                    self.show_process("Inheriting for ... {}".format(cl))
                    self.inheritProperties(cl, [x[-1] for x in temp_searchFormats])
                    
        codebase_result = {}
        if len(ans) > 0:
            #this maps the related que to data
            dataMap = [self.mapStrings(data, que[x]) for x in range(len(que))]
            
            #this gets the common que in ans   
            commonFormatQue = [self.getCommonFormat(que[x],ans[x]) for x in range(len(que))]
            
            #this gets the common anns in que 
            commonFormatAns = [self.getCommonFormat(ans[x],que[x]) for x in range(len(ans))]

            #this uses the datamap to turn the individual que to look as that of data
            searchFormats = [self.getCommonFormat(que[x],data) for x in range(len(que))]
            
            #this uses the datamap to turn the individual ans to look as that of data
            outputFormats = [self.formatOutput(dataMap[x],commonFormatAns[x]) for x in range(len(commonFormatAns))]

            #keep the originals
            original_ans = ans.copy()
            original_commonFormatAns = commonFormatAns.copy()
            original_outputFormats = outputFormats.copy()
            
            #to see if constant vars are availabel
            is_same = all(sorted(commonFormatAns[0][-1]) == sorted(x[-1]) for x in commonFormatAns)
            
            allclasses = []
            for i in range(len(que)):
                allclasses.append([])
                if outputFormats[i] == "[var]":
                    #ref1 will be the common betw the que and ans
                    ref  = self.getMRelated(commonFormatAns[i][-1])

                    #ref will be the common betwn the data and the que
                    ref2 = self.getMRelated(searchFormats[i][-1])
                    chk = self.sort_dict(self.intersect(ref2, ref))
                    
                    if len(chk) > 0 and self.formatOutput(dataMap[i],commonFormatAns[i]) != "[var]" and self.formatOutput(dataMap[i],commonFormatAns[i]) in outputFormats:
                        ans[i] = chk[0][0]
                        commonFormatQue[i] = self.getCommonFormat(que[i],ans[i])
                        commonFormatAns[i] = self.getCommonFormat(ans[i],que[i])
                        outputFormats[i] = self.formatOutput(dataMap[i],commonFormatAns[i])            
                self.show_process("\nque = {}\nans = {}\ndatamap = {}\ncommonFormatQue = {}\ncommonFormatAns = {}\noutputFormat = {}\nsearchFormat = {}\n".format(que[i], ans[i], dataMap[i], commonFormatQue[i], commonFormatAns[i], outputFormats[i], searchFormats[i]))
            
            outputFormats_infl = {x:outputFormats.count(x)/len(outputFormats) for x in set(outputFormats)}
            
            ansval = {}
            
            pd = self.sort_dict(outputFormats_infl)
            
            infl = {x:outputFormats_infl[x] for x in outputFormats_infl if outputFormats_infl[x] > 0}
            
            xx = {}
##------------------------this classifies the output formats and studies each to get reply---------------------------------------------------------------------
            output_classes = {x:[i for i in range(len(outputFormats)) if x == outputFormats[i]] for x in set(outputFormats)}
##            for x in output_classes:
##                for b in output_classes[x]:
##                    outputFormats_infl[x] *= related[que[b]]
            self.show_process("output_classes = {}".format(output_classes))
            self.show_process("outputFormats_infl = {}".format(pd))
            
            for cl in output_classes:
                output_class = output_classes[cl]

                temp_que = [que[y] for y in output_class]

                temp_ans = [ans[y] for y in output_class]
                
                #this maps the related que to data
                temp_dataMap = [dataMap[y] for y in output_class]
                
                #this gets the common que in ans   
                temp_commonFormatQue = [commonFormatQue[y] for y in output_class]
                
                #this gets the common anns in que 
                temp_commonFormatAns = [commonFormatAns[y] for y in output_class]

                #this uses the datamap to turn the individual que to look as that of data
                temp_searchFormats = [searchFormats[y] for y in output_class]
                
                #this uses the datamap to turn the individual ans to look as that of data
                temp_outputFormats = [outputFormats[y] for y in output_class]

                #to see if constant vars are availabel
                temp_is_same = all(sorted(temp_commonFormatAns[0][-1]) == sorted(x[-1]) for x in temp_commonFormatAns)

                #check for activation
                temp_search_memory_li = [self.memory.index(temp_ans[x]) < self.memory.index(temp_que[x]) for x in range(len(temp_ans))]
                temp_search_memory = max([temp_search_memory_li.count(x) for x in set(temp_search_memory_li)]) == temp_search_memory_li.count(True)

                self.show_process("search_memory = {}".format(temp_search_memory))
                
                ##this is supposed to search the code base to check for inherent capabilities like addition
                rel, clss = self.search_codebase(data, que, ans)
                
                rq = [x for x in rel]
                dm =  [self.mapStrings(data, x) for x in rel]
                rep = [self.formatOutput(dm[x],(rq[x],[])) for x in range(len(rq))]

                v_infl = {}
                if len(ans) > 1:
                    v_infl = {x:rep.count(x)/len(rep) for x in set(rep)}
                    self.process_event(v_infl, clss)
                res = {}
                for cpb in v_infl:
                    r = self.try_process(cpb)
                    if r != None:
                        if r in res:
                            res[r] += v_infl[cpb]
                        else:
                            res.setdefault(r, v_infl[cpb])
                            
##                for i in range(len(rel)):
##                    print("rq",rq[i], "rep",rep[i])

                print(res)
                allclasses = []
                for i in range(len(temp_ans)):
                    allclasses.append([])
            
                    if len(temp_commonFormatAns[i][-1]) > 0 and temp_is_same == False:
                        if cl == "my name is [var]":print(temp_commonFormatAns[i][-1])
                        allclasses[-1].extend(self.getAllClasses(temp_commonFormatAns[i][-1]))
                    else:
                        if cl == "my name is [var]":print(temp_commonFormatAns[i][0].replace("[var]",""))
                        allclasses[-1].extend(self.getAllClasses([temp_commonFormatAns[i][0].replace("[var]","")]))

##                for r in allclasses: print(r)
                temp_genScore = self.getScore(allclasses)
##                self.show_process("temp_score = {}".format(temp_genScore))
                
                temp_outputFormat = cl
                temp_xx = {}
                if all(data == x for x in temp_que):
                    temp_xx = ans.copy()
                    temp_xx = {x:(temp_xx.count(x)/len(temp_xx))*outputFormats_infl[temp_outputFormat] for x in temp_xx}
                    
                elif len(ans) > 1 and all(temp_outputFormat == x for x in temp_outputFormats):
                    if all(sorted(temp_commonFormatAns[0][-1]) == sorted(x[-1]) for x in temp_commonFormatAns):
                        st = temp_outputFormat
                        for x in temp_commonFormatAns[0][-1]:
                            st = st.replace("[var]", x, 1)
                        temp_xx = {st:outputFormats_infl[temp_outputFormat]}
                        
                    else:
                        if temp_search_memory:
                            rel = self.sort_dict(self.getRelated(temp_outputFormat, "",  0.1))
                            rel = [x[0] for x in rel if self.formatOutput(self.mapStrings(data, data), self.getCommonFormat(x[0], data)) == temp_outputFormat]
##                            print(rel)
##                            m = self.sort_last(rel)
##                            if m != -1 :
##                                temp_xx = {rel[m]: 1.0}
##                            else:
##                                temp_xx = {}
                            temp_xx = {x:outputFormats_infl[temp_outputFormat] for x in rel}
                
                temp_xx_infl = self.getClassScore(temp_xx, temp_genScore, data, temp_is_same)

                self.show_process("temp_xx_infl = {}".format(temp_xx_infl))      
                pd = self.sort_dict(temp_xx_infl)
                #print(pd)
                temp_xx = {x:temp_xx[x] for x in temp_xx_infl if temp_xx_infl[x] >= pd[0][-1]*0.9 and temp_xx_infl[x] > 0}
                
                for x in res:
                    if x in codebase_result:
                        codebase_result[x] += res[x]*outputFormats_infl[temp_outputFormat]
                    else:
                        codebase_result.setdefault(x, res[x]*outputFormats_infl[temp_outputFormat])

                print("xxxxx", cl, temp_xx)

                for val in temp_xx:
                    if val in xx:
                        xx[val] *= temp_xx[val]
                    else:
                        xx.setdefault(val, temp_xx[val])
                        
            xx.update(codebase_result)
            for pred in xx:
                if pred in ansval:
                    ansval[pred] += 1
                else:
                    ansval.setdefault(pred, 1)
                    
            ansval = {x:ansval[x]*xx[x] for x in ansval}
            
            pd = self.sort_dict(ansval)
            ansval = {x:ansval[x] for x in ansval if ansval[x] >= pd[0][-1]*0.75}
            #if its a single data with multiple replies        
            if all(data == x for x in que):     #get the best replied one
                if all(data == x for x in que): ansval = {x:ansval[x]*(int(self.read_datafreq(data)[self.read_data(data).index(x)])/int(self.readfreq(data))) for x in ansval if x in self.read_data(data)}

            self.show_process("suggested ans = {}".format(xx))
            self.show_process("ansval = {}".format(ansval))
            
            self.show_process()
                
            if len(ansval) > 0:
                pd = self.sort_dict(ansval)
                predicted_ans = {pd[0][0]:pd[0][-1]}
                c = 1
                while c < len(pd) and pd[0][-1] == pd[c][-1]:
                    predicted_ans.setdefault(pd[c][0], pd[c][-1])
                    c += 1
                self.show_process("predicted reply = {}".format(predicted_ans))
                
                
            else:
                return ""

    def save2memory(self, data, artificial=0):
        if data in self.memory:
            if artificial == 0: self.freq[self.memory.index(data)] = str(int(self.freq[self.memory.index(data)]) + 1)
        else:
            self.memory.append(data)
            self.saveInput(data)
            if artificial == 0: self.freq.append("1")
            else: self.freq.append("0")
        self.save()
		
    def save2codebase(self, cmd, value):
        self.codebase.setdefault(cmd, value)
        self.save()

    def toogle_state(self, var):
        if var:
            var = False
        else:
            var = True

    def event_process(self, data):
        for ev in self.events:
##            print('ev = ', ev, len(ev), '\nclasses = ', str(self.getAllClasses([data])), len(str(self.getAllClasses([data]))),  self.is_in(ev, str(self.getAllClasses([data]))))
##            input()
            if self.is_in(ev, str(self.getAllClasses([data]))):
                return "Yes"
