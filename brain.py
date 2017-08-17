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
            print(self.runCommand(data))
        
        else:
            if data not in ["~","/"]:
                self.save2memory(data)

##            print(self.context[-1], self.reply)
            if self.context[-1] == "/" and self.reply == True:
                r = self.confirm_event(data, self.context[-3])
##                print(r)
                if r == False:
##                    print(2, self.context[-5:], data, self.context[-2])
                    self.setReply(self.context[-3], data)
                    if self.context[-1] == data:
                        self.context.pop()
                            
##            print(self.context[-5:])
            self.setContext(data)
##            print(self.context[-5:])
            if len(self.context) > 2:
##                if self.context[-2] not in ["/", "~"] and data not in ["/", "~"]: self.setReply(self.context[-2], data)
                if(self.reply == True and self.context[-2] == data):
                    self.confirm_event(data, self.context[-3])

                if self.context[-2] != "/" and self.reply == False:
                    r = self.confirm_event(data, self.context[-2])
##                    print(r)
                    if r == False:
##                        print(1, self.context[-2], data)
                        self.setReply(self.context[-2], data)
                        if self.context[-2] == data:
                            self.context.pop()
                            
##                print(self.context[-5:])
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
            if not self.learning:
                #to compare present data with previous data and lock relation as event
                if len(self.context) > 1:
                    if self.context[-2] not in ["~", "/"]:
                        self.pre_process(self.context[-2], data)
                    else:
                        if self.context[-2] == "/" and len(self.context) > 3:
                            self.pre_process(self.context[-4], data)
                print('... pre processed')
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
        rel = self.getRelated(data, " ", ((len(data.split())/2)**-1)/2, strict=True)
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

        que, ans, qna_infl = self.getQueAns(relatedlist)
        
        return que, ans, qna_infl
    
    def pre_process(self, data, data_after):
        print('pre processing... data = {}, data_after = {}'.format(data, data_after))
        que, ans, qna_infl, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, output_classes = self.get_classes(data)

        outf = self.formatOutput(self.mapStrings(data, data), self.getCommonFormat(data_after, data))
        
        if outf not in output_classes:
            return
        
        output_class = output_classes[outf]
        temp_qna_infl = [qna_infl[y] for y in output_class]
        temp_que = [que[y] for y in output_class]
   
        #this uses the datamap to turn the individual que to look as that of data
        temp_searchFormats = [searchFormats[y] for y in output_class]

        sf = [z[0] for z in temp_searchFormats]
        key = ' '
        
        if len(sf) < 1 and sum(temp_qna_infl) < 3:
            return
    
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
        print(k, len(sf))
        if len(k) > 1:
            k = self.sort_dict(k)
            comp = int(k[0][-1])/len(sf)
            print(comp)
            if comp > 0.75:
                key = k[0][0]
            print('k', key)

        ac = []
        for y in temp_que:
            ac.append(self.getAllClasses([y]))
        ac = sorted(self.getCommon(ac))
        data_class = sorted(ac)
        
        cdb = 0
        r = self.try_process(data)
        if r == data_after:
            cdb = 1
##        (cl.replace('[var]', y[0]), y[-1], data, temp_searchFormat_max, x, common_cls)
        print('class', outf, cdb)
        if str(sorted(data_class)) not in self.events and self.confirmed_event == False:
            if cdb == 1:
                self.set_event(data, sorted(data_class), str(cdb), '0/0', data, key)
            else:
                related_que = self.find_answer(data, data_after)
                
                print('rel', related_que[:10])
                if related_que == []:
                    self.generateEvent(data, data_after, key, outf)
                        
                for x in related_que:
                    r = self.try_process(x)
                    if r != None:
                        self.set_event(data, sorted(data_class), '1', '0/0', x, key)
                        print(data, '----and----', x, 'here-------------------------------------')

                    else:
                        self.generateEvent(data, data_after, key, outf)
                        
    def generateEvent(self, data, data_after, key, outf):
        for x in self.last_predicted_list:
            print(x[:-1])
            if data_after == x[0]:
                print('data_rel', x[3])
                relxx = self.getQueRelated(x[3])

                print('relxx', self.sort_dict(relxx)[:10])
                ac = []
                for y in relxx:
                    ac.append(self.getAllClasses([y]))
                ac = sorted(self.getCommon(ac))
                data_class = sorted(ac)
                
                self.set_event(data, data_class, '0', '0/0', outf, key, x[-1])
                print(data, '=====>', outf, 'here-------------------------------------')
                
    def isset(self, data, que):
        predicted_list = {}
        if all(data == x for x in que):     #get the best replied one
            print([(x, int(self.read_datafreq(data)[self.read_data(data).index(x)]), int(self.readfreq(data))) for x in predicted_list if x in self.read_data(data)])
            if all(data == x for x in que): predicted_list = {x:predicted_list[x]*(int(self.read_datafreq(data)[self.read_data(data).index(x)])/int(self.readfreq(data))) for x in predicted_list if x in self.read_data(data)}
            return predicted_list

    def get_classes(self, data):
        self.show_process("data = {}".format(data))
        que, ans, qna_infl = self.getQnA(data)

        dataMap = commonFormatQue = commonFormatAns = searchFormats = outputFormats = output_classes = []
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

##------------------------this classifies the output formats and studies each to get reply---------------------------------------------------------------------
            output_classes = {x:[i for i in range(len(outputFormats)) if x == outputFormats[i]] for x in set(outputFormats)}
            
            self.show_process("output_classes = {}\n".format(output_classes))
        return que, ans, qna_infl, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, output_classes
        
    def compute(self,data):
            que, ans, qna_infl, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, output_classes = self.get_classes(data)
##            self.show_process("outputFormats_infl = {}".format(pd))
            
            #if its a single data with multiple replies        
            r = self.isset(data, que)
            if r != None:
                return self.Filter(r)
            
            #instantiating variables
            predicted_list = []     #this is the predicted ans
            predicted_no = []
            for cl in output_classes:
                output_class = output_classes[cl]

                temp_qna_infl = [qna_infl[y] for y in output_class]
                
                if sum(temp_qna_infl) < 2:
                    continue
                
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
                temp_searchFormat_max = self.getModelMax([x[0] for x in temp_searchFormats])
                
                #this uses the datamap to turn the individual ans to look as that of data
                temp_outputFormats = [outputFormats[y] for y in output_class]

                #output format
                temp_outputFormat = cl

                #to see if constant vars are availabel
                temp_is_same = all(sorted(temp_commonFormatAns[0][-1]) == sorted(x[-1]) for x in temp_commonFormatAns)

                #check for activation
                temp_search_memory_li = [self.memory.index(temp_ans[x]) < self.memory.index(temp_que[x]) for x in range(len(temp_ans))]
                temp_search_memory = max([temp_search_memory_li.count(x) for x in set(temp_search_memory_li)]) == temp_search_memory_li.count(True)

                self.show_process('class: {}'.format(cl))
                self.show_process("search_memory = {}".format(temp_search_memory))
                
                common_classes = []
                
                cl_db = dict([])      #variable to hold the model of the ans
                
                class_infl = sum(temp_qna_infl)/sum(qna_infl)
                if sum(temp_qna_infl) > 2:
##--------------------------this is to derive the model to fetch the ans------------------------------
                    ans_length = len(temp_ans)
                    li = temp_ans.copy()
                    li.insert(len(li), data)
                    if cl == '[var]':
                        for vx in li:
                            vcl = self.getAllClasses([vx])
                            for vxn in vcl:
                                if vxn != '[var]' and vxn in cl_db:
                                    cl_db[vxn] += 1
                                else:
                                    if vxn != '[var]': cl_db.setdefault(vxn, 1)                        
                        cl_db = {x:(cl_db[x]/len(temp_ans))*class_infl for x in cl_db if cl_db[x] > 1}
                        
                    else:
                        cl_db = {cl:1.0*class_infl}
                
                for cx in temp_commonFormatAns:
                    for ct in range(cx[0].count('[var]')):  
##                        print(cx[-1][ct])
                        vcl = self.orClass(cx[-1][ct], sep=" ")
                        common_classes.append(vcl)

                temp_genScore = self.getScore(common_classes, strict=False)
                self.show_process("temp_score = {}".format(self.sort_dict(temp_genScore)[:10]))
                if '[var] is a place' in temp_genScore:
                    print('[var] is a place', temp_genScore['[var] is a place'])

                common_cls = [] # this all the classification of the answer for creating events through predicted_list
                if len(temp_genScore) > 0:
                    temp_genScore_li = self.sort_dict(temp_genScore)
                    common_cls = [x for x in temp_genScore_li if x[-1] == temp_genScore_li[0][-1]]

                print('common_cls = ', common_cls[:10])    
                self.show_process("class list_score = {}".format(self.sort_dict(cl_db)[:10]))
                self.show_process('score {} {} {} {}'.format(temp_qna_infl, sum(temp_qna_infl), sum(qna_infl), sum(temp_qna_infl)/sum(qna_infl)))
                cmps = []
##                print('cl_db', self.sort_dict(cl_db))
                if len(cl_db) > 0:
                    for x in cl_db:
##                        print('\n working on',x)
                        ret = self.matchAns(data, x)
                        for y in ret:
                            cmps.append((cl.replace('[var]', y[0]), y[-1], data, temp_searchFormat_max, x, common_cls))
                            
##                print(self.sort_dict(cmps)[:10])
                self.show_process()
                pr = self.getClassScore2({x[0]:'' for x in cmps}, temp_genScore, data=False, is_same=False, sep= " ")
                if pr != []:
                    cmps = [x for x in cmps if x[0] in pr]
                print('pr', pr)
                predicted_list.extend(cmps)
                predicted_no.extend([x[0] for x in cmps])

##            self.show_process("suggested ans = {}".format(predicted_list))
            m = 0
            for x in set(predicted_no):
                c = predicted_no.count(x)
                if c > m:
                    m = c
            predicted_list = [x for x in predicted_list if len(predicted_no) > 0 and predicted_no.count(x[0]) == m]
            self.predicted_list = predicted_list
            self.show_process("final suggested ans = {}".format([x[0] for x in predicted_list]))
            return self.Filter(predicted_list)
        
    def matchAns(self, data, x):
        cmp = self.getRelated(x, sep=" ", treshold=(len(x.split())**-1)/2, strict=True)
        vs = [xx for xx in cmp if self.percentage_similarity(data, x, sep=" ") > 0.1]
        li = []
        for vx in vs:
            i = self.memory.index(vx)
            dm =  self.mapStrings(x, vx)
##                            -----------------this is to get the corresponding [var] value in data map---------------------
            rep = " "+" ".join([xy[-1].replace((' '+xy[0]+' ').replace(' [var] ', '', 1).strip(), '', 1).strip() for xy in dm if ' [var] ' in ' '+xy[0]+' ']) +" "
            if " "+x.replace('[var]', '').strip()+" " in " "+vx+" " and rep not in " "+data+" " and len(rep.strip()) > 0 and rep in " "+vx+" ":
                self.show_process("{} {} {}".format(" "+x.replace('[var]', '').strip()+" ",'---in----',  " "+vx+" "))
                li.append((rep.strip(), i))
        return li
    
    def Filter(self, predicted):
        if len(predicted) > 0:
            i = 0
            for x in predicted:
                if x[1] > i:
                    reply = x[0]
                    i = x[1]
            return reply
    
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
                self.show_process('\nsearching ... '+self.events_id[x])
                event_id = False
                key = self.read_event(x, 'key')
##                print('key', key)
                if len(key.strip()) > 0:
                    if " "+key+" " in " "+data+" ":
                        go = True
                    else:
                        go = False
                else:
                    if self.getRelation(data, self.events_id[x]) < 0.01:
                        go = False
                    else:
                        go = True
##                print('before go')
                if go and self.events_id[x] in self.getRelated(data, sep=" ", treshold=0.01, strict=True):
                    event_id = self.events_id[x]
##                    print('after go')
                    if event_id != False and self.is_in(ev, str(cls), 10e-01):
                        self.show_process('matching events = {} for {}'.format(event_id, data))
                        dm =  self.mapStrings(data, event_id)
                        rep = self.formatOutput(dm,(self.read_event(x, 'format'),[]))
##                        print(dm, self.read_event(x, 'format'), rep)
                        
                        if self.read_event(x, 'codebase') == '1':
                            rep = self.try_process(rep)
                            
                        else:
                            print(data, self.read_event(x, 'format'))
                            rep = self.matchAns(data, self.read_event(x, 'format'))
                            print(rep)
                        expected_ans = (x, data, rep)
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
                            self.show_process('cmd = {}, expected ans = {}, confidence = {}'.format(rep, expected_ans[-1], confidence))

                            return expected_ans[-1]

                        
