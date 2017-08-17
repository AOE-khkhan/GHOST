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

    def prepare(self, data):
        if data.startswith("?"):
            return self.runCommand(data)
        else:
            if data not in ["~","/"]:
                self.save2memory(data)

            if self.context[-1] == "/" and self.reply == True:
                self.confirm_event(data, self.context[-3])
                
            self.setContext(data)

            if len(self.context) > 2:
                self.setReply(self.context[-2], data)
                if self.reply == False: self.confirm_event(data, self.context[-2])
                if(self.reply == True and self.context[-2] == data): self.confirm_event(data, self.context[-3])
                
            if len(self.context) > 3 and self.context[-2] == "~" and not self.learning:
                self.setReply(self.context[-3], data)
                if self.reply == False: self.confirm_event(data, self.context[-3])
##                if(self.reply == True and self.context[-2] == data): self.confirm_event(data, self.context[-3])
                
            if data not in ["~","/"]:
                self.expected_ans = []
                
            if data == "~":
                self.switchSource()
                self.manageSource()
            return 'prepared'

    def run_computation(self, data):
        if data not in ["~","/"]:
##            for x in data.split(" "):
##                self.save2memory(x, 1)
##                self.generateArtificialData(x)
            if not self.learning:
                #to compare present data with previous data and lock relation as event
                if len(self.context) > 1:
                    if self.context[-2] not in ["~", "/"]:
                        self.pre_process(self.context[-2], data)
                    else:
                        if self.context[-2] == "/" and len(self.context) > 3:
                            self.pre_process(self.context[-4], data)

                r = self.event_process(data)
                if r == None:
                    return (self.compute(data), 1)
                else:
                    return (r, 1)
                
    def analyse(self, data):
        r = self.prepare(data)
        if r == 'prepared':
            return self.run_computation(data)
        else:
            return r

    def getQueRelated(self, data):
        rel = self.getRelated(data, " ", 0.1, strict=True)
        related = rel.copy()             
        for x in rel:
            if len(x.split()) != len(data.split()):
                related[x] *= 0.5
        pd = self.sort_dict(related)

        m = 0
        if len(pd) > 0:
            if pd[0][0] == data and len(pd) > 1: m = 1

            related = {x:related[x] for x in related if related[x] >= pd[m][-1]*0.5}
            return related
        else:
            return {}

    def inheriter(self, data, rel, searchFormats):
        rel = relatedlist.copy()
        if data in rel: rel.remove(data)
        
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

##        for cls in [searchFormat_classes, new_searchFormat_classes]:
##            for cl in cls:
##                temp_searchFormats = [searchFormats[i] for i in cls[cl]]
##                
##                #this inherits properties from the related values to generate more artificial data 
##                if len(temp_searchFormats) > 1 and all([temp_searchFormats[0][0] in a for a in temp_searchFormats]):
##                    self.show_process("Inheriting for ... {}".format(cl))
##                    self.inheritProperties(cl, [x[-1] for x in temp_searchFormats])

    def trimModels(self, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats):
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

        return dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats

    def getQnA(self, data):
        related = self.getQueRelated(data)
        relatedlist = [x for x in related]
        self.show_process('related = {}'.format(relatedlist[:10]))    

        que, ans = self.getQueAns(relatedlist)
        
        return que, ans
    
    def pre_process(self, data, data_after):
##        print('pre processing... data = {}, data_after = {}'.format(data, data_after))
        related_que = self.find_answer(data, data_after)
        
        for x in related_que:
            relxx = self.getRelated(data, sep=" ", treshold=0.3, strict=True)
            
##            print('relxx', x, relxx)
            ac = []
            for y in relxx:
                ac.append(self.getAllClasses([y]))
            ac = sorted(self.getCommon(ac))
            data_class = sorted(ac)

            if str(sorted(data_class)) not in self.events or self.confirmed_event == False:
                self.set_event(data, sorted(data_class), '0', '0/0', x)
##        print('... pre processed')
                
    def anon(self):
        que, ans = self.getQnA(data)
        dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats = self.generateModelers(data, que, ans)
        
        searchFormats = [x[0] for x in searchFormats]
        
        #this gets the maximum search format
        searchFormat = self.getModelMax(searchFormats)
        self.show_process("\nsearchFormat = {}".format(searchFormat))

        ql = {}
        for q in que:
            vq = q
            for qx in searchFormat.split():
                vq = vq.replace(qx, "", 1)
            ql.setdefault(q, vq)

        tq = []
        ta = []
        tql = []
        for qn in range(len(que)):
            if que[qn] in ql and searchFormats[qn] == searchFormat:
                tql.append(ql[que[qn]])
                tq.append(que[qn])
                ta.append(ans[qn])
            ql.pop(que[qn])
                
        ##this is supposed to search the code base to check for inherent capabilities like addition( for a sub class)
        print(searchFormats)
        events = self.search_codebase(data, tq, ta)
            
        print(tq)
        print([x[0] for x in events])
        input()
        
        if len(ta) > 1:
            self.process_event(events, 1)
##                    print('\n----ans here', self.run_computation(vq))
                    
    def compute(self,data):
        self.show_process("data = {}".format(data))
        que, ans = self.getQnA(data)
        
##-----------------------------------------this is for inheritance of properties when no ans---------------------------------------------------------------------
##        self.inheriter()
                    
        if len(ans) > 0:
            dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats = self.generateModelers(data, que, ans)

            #keep the originals
            original_ans = ans.copy()
            original_commonFormatAns = commonFormatAns.copy()
            original_outputFormats = outputFormats.copy()

            
            #to see if constant vars are availabel
            is_same = all(sorted(commonFormatAns[0][-1]) == sorted(x[-1]) for x in commonFormatAns)

            for i in range(len(que)):
                self.show_process("\nque = {}\nans = {}\ndatamap = {}\ncommonFormatQue = {}\ncommonFormatAns = {}\noutputFormat = {}\nsearchFormat = {}\n".format(que[i], ans[i], dataMap[i], commonFormatQue[i], commonFormatAns[i], outputFormats[i], searchFormats[i]))

##            dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats = self.trimModels(dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats)

            outputFormats_infl = {x:outputFormats.count(x)/len(outputFormats) for x in set(outputFormats)}
            pd = self.sort_dict(outputFormats_infl)
            infl = {x:outputFormats_infl[x] for x in outputFormats_infl if outputFormats_infl[x] > 0}

            ansval = {}
            xx = {}
##------------------------this classifies the output formats and studies each to get reply---------------------------------------------------------------------
            output_classes = {x:[i for i in range(len(outputFormats)) if x == outputFormats[i]] for x in set(outputFormats)}
            
            self.show_process("output_classes = {}".format(output_classes))
##            self.show_process("outputFormats_infl = {}".format(pd))
            
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
                self.show_process('class: {}'.format(cl))
                sf = [z[0] for z in temp_searchFormats]
                key = ' '

                if len(sf) > 0 and len(temp_ans) > 1:
                    sfl = list(set(sf))
                    sfi = {x:sf.count(x) for x in sfl}
                    k = {}
                    
                    for xk in sfl:
                        for xxk in xk.split():
                            if xxk != '[var]' and xxk not in k:
                                for xkx in sfl:
                                    if xxk in xkx:
                                        if xxk in k:
                                            k[xxk] += sfi[xkx]
                                        else:
                                            k.setdefault(xxk, sfi[xkx])
##                    print(k, len(sf))
                    if len(k) > 1:
                        k = self.sort_dict(k)
                        comp = int(k[0][-1])/len(sf)
##                        print(comp)
                        if comp > 0.9:
                            key = k[0][0]
##                    print('k', key)
                
                if len(temp_ans) > 1:
                    ##this is supposed to search the code base to check for inherent capabilities like addition
                    events = self.search_codebase(data, temp_que, temp_ans)

                    self.process_event(events, 1, key)

                for vx in temp_ans:
##                    print(vx, self.getCommon([[x for x in self.getRelated(y, sep=" ", treshold=0.1, strict=True)] for y in [data, vx]]))                    
                    pass
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
                self.show_process("temp_score = {}".format(temp_genScore))

                #output format
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
                            temp_xx = {x:outputFormats_infl[temp_outputFormat] for x in rel}
                
                temp_xx_infl = self.getClassScore(temp_xx, temp_genScore, data, temp_is_same)

                self.show_process("temp_xx_infl = {}".format(temp_xx_infl))      
                pd = self.sort_dict(temp_xx_infl)
                #print(pd)
                temp_xx = {x:temp_xx[x] for x in temp_xx_infl if temp_xx_infl[x] >= pd[0][-1]*0.9 and temp_xx_infl[x] > 0}
                
                
##                print("xxxxx", cl, temp_xx)

                for val in temp_xx:
                    if val in xx:
                        xx[val] *= temp_xx[val]
                    else:
                        xx.setdefault(val, temp_xx[val])
                        
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
                print([(x, int(self.read_datafreq(data)[self.read_data(data).index(x)]), int(self.readfreq(data))) for x in ansval if x in self.read_data(data)])
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
                
                print("predicted reply = {}".format(predicted_ans))
                r = self.sort_dict(predicted_ans)
                if len(r) > 0:
                    if r[0][-1] >= 0.5:
                        return r[0][0]
                    else:
                        return None
            else:
                return None

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
        self.saveData("codebase", [str(len(self.codebase))+"\n"])
        
    def toogle_state(self, var):
        if var:
            var = False
        else:
            var = True

    def event_process(self, data=None, cls=None):
        if cls == None:
            cls = self.getAllClasses([data])

        for ev in self.events:
            for x in self.events[ev]:
                print('\nsearching ... '+self.events_id[x])
                event_id = False
                key = self.read_event(x, 'key')
                if len(key.strip()) > 0:
                    if " "+key+" " in " "+data+" ":
                        go = True
                    else:
                        go = False
                else:
                    go = False
                if go and self.events_id[x] in self.getRelated(data, sep=" ", strict=True):
                    event_id = self.events_id[x]

                    if event_id != False and self.is_in(ev, str(cls), 10e-01):
                        print('matching events = {} for {}'.format(event_id, data))
                        dm =  self.mapStrings(data, event_id)
                        rep = self.formatOutput(dm,(self.read_event(x, 'format'),[]))
        ##                print(dm, self.read_event(event_id, 'format'), rep)
                        expected_ans = (x, data, self.try_process(rep))

                        self.set_expected_ans(expected_ans, self.new_expected_ans)
                        
                        confidence = self.read_event(x, 'confidence')
                        confidence = self.try_process(confidence)
                        
                        try:
                            confidence = float(confidence)
                        except Exception as e:
                            confidence = 0
                            
                        if confidence == None:
                            confidence = 0
        ##                print(dm, self.read_event(event_id, 'format'), rep, confidence)
                        if expected_ans[-1] != None: print('cmd = {}, expected ans = {}, confidence = {}'.format(rep, expected_ans[-1], confidence))
                        if expected_ans[-1] != None and dm != [] and confidence >= 0.5:
                            print('cmd = {}, expected ans = {}, confidence = {}'.format(rep, expected_ans[-1], confidence))

                            self.setContext(expected_ans[-1])
                            self.setReply(data, expected_ans[-1])
                            return expected_ans[-1]
            
