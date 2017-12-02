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
                for x in data.split():
                    self.save2memory(x, 1)
                self.save2memory(data)

            if self.reply == True:
                if self.context[-1] == "/":
                    confidence = (0,1)
                else:
                    confidence = (1,1)
                if self.used_event != False:
                    self.increase_confidence(self.used_event, confidence)
                    self.used_event = False

                if self.context[-3] not in ["~","/"] and data not in ["~","/"]:
                    r = self.confirm_event(data, self.context[-3])
                    if r == False:
                        self.setReply(self.context[-3], data)
                        if self.context[-1] == data:
                            self.context.pop()
                            
            self.setContext(data)

            if len(self.context) > 2:
##                if self.context[-2] not in ["/", "~"] and data not in ["/", "~"]: self.setReply(self.context[-2], data)
                if(data not in ["~","/"] and self.context[-3] not in ["~","/"] and self.reply == True and self.context[-2] == data):
                    self.confirm_event(data, self.context[-3])

                if self.context[-2] not in ["~","/"] and self.reply == False:
                    r = self.confirm_event(data, self.context[-2])
##                    print(r)
                    if r == False and data not in ["~","/"]:
##                        print(1, self.context[-2], data)
                        self.setReply(self.context[-2], data, True)
                        if self.context[-2] == data:
                            self.context.pop()
                            
##                print(self.context[-5:])
            if len(self.context) > 3 and self.context[-2] in ["~","/"] and data not in ["~","/"] and not self.learning:
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
                self.show_process('... pre processed')
                r = self.event_process(data)
                if r == None:
                    return (self.compute(data), 1)
                else:
                    return (self.Filter([(r, 1)]), 1)
                
    def analyse(self, data):
        r = self.prepare(data)
        if r == 'prepared':
            return self.run_computation(data)
        else:
            return r

    def getQueRelated(self, data):
        if len(data.split()) > 0:
            rel = self.getRelated(data, " ", (len(data.split())**-1)/2, strict=True)
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
                    outputFormats[i] = self.formatOutput(dataMap[i],commonFormatAns[i]).strip()          
            self.show_process("\nque = {}\nans = {}\ndatamap = {}\ncommonFormatQue = {}\ncommonFormatAns = {}\noutputFormat = {}\nsearchFormat = {}\n".format(que[i], ans[i], dataMap[i], commonFormatQue[i], commonFormatAns[i], outputFormats[i], searchFormats[i]))

        return dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats

    def getQnA(self, data):
        related = self.getQueRelated(data)
        relatedlist = [x for x in related]
        self.show_process('related = {}, count = {}'.format(relatedlist[:10], len(relatedlist)))    

        que, ans, qna_infl = self.getQueAns(relatedlist)
        
        return que, ans, qna_infl
    
    def pre_process(self, data, data_after):
        self.show_process('\npre processing...')
        self.show_process('data = {}, data_after = {}'.format(data, data_after))
        que, ans, qna_infl, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, output_classes = self.get_classes(data)

        outf = self.formatOutput(self.mapStrings(data, data), self.getCommonFormat(data_after, data))
        
        if outf not in output_classes:
            return
        
        output_class = output_classes[outf]
        temp_qna_infl = [qna_infl[y] for y in output_class]
        temp_que = [que[y] for y in output_class]
        temp_ans = [ans[y] for y in output_class]
        
        #this uses the datamap to turn the individual que to look as that of data
        temp_searchFormats = [searchFormats[y] for y in output_class]

        sf = [z[0] for z in temp_searchFormats]
        self.show_process("sf = {}".format(sf)) 
        key = ' '
        
        if len(sf) < 1 and sum(temp_qna_infl) < 3:
            return
        
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
        self.show_process('class = {}'.format(outf))
        if str(sorted(data_class)) not in self.events and self.confirmed_event == False:
            if cdb == 1:
                if self.confirmed_event == False:
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

                    k = {x:k[x] for x in k if k[x] < sum(sfi.values())}
                    for x in k:
                        if x in self.memory:
                            f = int(self.memory[x]["ifreq"])
                            if f > 1:
                                k[x] *= (1 - (f**-1))

                    self.show_process("key_list = {}".format(k))
                    sfl = sum([k[x] for x in k])
                    if sfl > 0:
                        k = self.sort_dict(k)

                        key = k[0][0]
                        self.show_process('key = {}'.format(key))

                    self.set_event(data, sorted(data_class), str(cdb), '0/0', data, key)
            else:
                related_que = self.find_answer(data, data_after)
                
                self.show_process('rel = {}'.format(related_que[:10]))
                if related_que == []:
                    self.generateEvent(data, data_after, key, outf)
                        
                for x in related_que:
                    dm =  self.mapStrings(data, x)
                    string = x
                    if any([vx[0] == vx[-1] for vx in dm]):
                        for vx in dm:
                            if vx[0] == vx[-1]:
                                string = string.replace(vx[0], "", 1)
                        key = self.formatOutput(dm,(string, []))
                    else:
                        string = " "+data+" "
                        for vx in x.split():
                            if " "+vx+" " in string:
                                string = string.replace(" "+vx+" ", " ", 1)
                        key = string.strip()
                    print(dm, string, key)
                    self.show_process('key = {}'.format(key))
                    r = self.try_process(x)
                    if r != None:
                        self.set_event(data, sorted(data_class), '1', '0/0', x, key)
                        
                    else:
                        self.generateEvent(data, data_after, key, outf)
                        
    def generateEvent(self, data, data_after, key, outf):
        for x in self.last_predicted_list:
            if data_after == x[0]:

                relxx = self.getQueRelated(x[3])

                ac = []
                for y in relxx:
                    ac.append(self.getAllClasses([y]))
                ac = sorted(self.getCommon(ac))
                data_class = sorted(ac)
                
                self.set_event(data, data_class, '0', '0/0', outf, key, x[-1])
                
    def isset(self, data, que, ans, cl, temp_searchFormat_max, common_cls):
        rev_session = self.session.copy()
        rev_session.reverse()
        length = len(self.session)

        predicted_list = {}
        if all(data == x for x in que):     #get the best replied one
            predicted_list = {x:(int(self.memory[data]["ans"][x])/int(self.memory[data]["freq"])) for x in ans if x in self.memory[data]["ans"]}

            predicted = []
            
            for x in predicted_list:
                if x in self.session:
                    i = length - 1 - rev_session.index(x)

                else:
                    i = -1
                predicted.append((cl.replace('[var]', x), i, data, temp_searchFormat_max, x, common_cls))
            return predicted

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
            output_classes = {x.strip():[i for i in range(len(outputFormats)) if x == outputFormats[i]] for x in set(outputFormats)}
            
            self.show_process("output_classes = {}\n".format(output_classes))
        return que, ans, qna_infl, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, output_classes
        
    def compute(self,data):
            que, ans, qna_infl, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, output_classes = self.get_classes(data)
##            self.show_process("outputFormats_infl = {}".format(pd))
            
            #instantiating variables
            predicted_list = []     #this is the predicted ans
            predicted_no = []
            for clx in output_classes.copy():
                cl = clx
                output_class = output_classes[clx]

                temp_que = [que[y] for y in output_class]
                
                temp_ans = [ans[y] for y in output_class]

                temp_qna_infl = [qna_infl[y] for y in output_class]
                
                if sum(temp_qna_infl) < 2:
                    continue

                else:
                    self.show_process('\nclass: {}'.format(cl))

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

                var_infl = {}
                for cfax in temp_commonFormatAns:
                    for cfa in cfax[-1]:
                        cfav = cfa+'`'+str(cfax[-1].index(cfa))
                        if cfav in var_infl:
                            var_infl[cfav] += 1

                        else:
                            var_infl.setdefault(cfav, 1)

                var_infl = {x:var_infl[x]/len(temp_commonFormatAns) for x in var_infl}
##                self.show_process('\npossible replacements = {}'.format(var_infl))
                
                var_pd = [x[0] for x in self.sort_dict(var_infl) if x[-1] >= 0.9]
                if len(var_pd) > 0:
                    var, ind = var_pd[0].split('`')
                    var_li = cl.split('[var]')
                    new_var = var_li.copy()
                    ind = int(ind)
                    for i in range(cl.split().count('[var]')):
                        if i == ind:
                            new_var[ind] += ' '+var
                            new_var[ind] = new_var[ind].strip()

                            if len(new_var)-1 > ind:
                                new_var[ind] += new_var[ind+1]
                                new_var.pop(ind+1)
                                new_var[ind] = new_var[ind].strip()
                            
                    if new_var != var_li:
                        new_cl = ' [var] '.join(new_var).strip()
                        output_classes.setdefault(new_cl, output_classes[cl])
                        cl = new_cl
                self.show_process('possible replacements = {}'.format(var_pd))
                self.show_process('this is a new class = {}\n'.format(cl))

                #output format
                temp_outputFormat = cl

                #to see if constant vars are availabel
                temp_is_same = all(sorted(temp_commonFormatAns[0][-1]) == sorted(x[-1]) for x in temp_commonFormatAns)

                #check for activation
                temp_search_memory_li = [self.session.index(temp_ans[x]) < self.session.index(temp_que[x]) for x in range(len(temp_ans))]
                
                self.show_process("search_memory_li = {}".format(temp_search_memory_li))
                temp_search_memory = max([temp_search_memory_li.count(x) for x in set(temp_search_memory_li)]) == temp_search_memory_li.count(True)

##                self.show_process("search_memory = {}".format(temp_search_memory))
                
                common_classes = []
                
                cl_db = dict([])      #variable to hold the model of the ans
                false_list = [] #used to store indexes cases which replies are already in memory
                
                class_infl = sum(temp_qna_infl)/sum(qna_infl)
                if sum(temp_qna_infl) > 1:
##--------------------------this is to derive the model to fetch the ans------------------------------
                    ix = 0
                    ali = []

                    if False in temp_search_memory_li and cl != '[var]':
                        for i in range(temp_search_memory_li.count(False)):
                            LI = temp_search_memory_li[ix+1:]

                            if False in LI:
                                if temp_search_memory_li.index(False) == 0 and 0 not in false_list:
                                    false_list.append(0)

                                ix += LI.index(False) + 1
                                false_list.append(ix)
                                vx = temp_searchFormats[ix][-1].copy()
                                
                                if len(vx) > 0:
                                    vx = vx[0]
                                else:
                                    continue
                                
                                ali.append((vx, temp_dataMap[ix], temp_ans[ix].replace(vx, '', 1).strip(), ix))

                        if ali == []:
                            ali.append((data, temp_dataMap[0], '', ix))
                            
                    xtemp_que = [que[y] for y in output_class if y in false_list]
                
                    xtemp_ans = [ans[y] for y in output_class if y in false_list]

                    xtemp_qna_infl = [qna_infl[y] for y in output_class if y in false_list]
                    
                    #this maps the related que to data
                    xtemp_dataMap = [dataMap[y] for y in output_class if y in false_list]
                    
                    #this gets the common que in ans   
                    xtemp_commonFormatQue = [commonFormatQue[y] for y in output_class if y in false_list]
                        
                    #this gets the common anns in que 
                    xtemp_commonFormatAns = [commonFormatAns[y] for y in output_class if y in false_list]

                    #this uses the datamap to turn the individual que to look as that of data
                    xtemp_searchFormats = [searchFormats[y] for y in output_class if y in false_list]
                    xtemp_searchFormat_max = self.getModelMax([x[0] for x in temp_searchFormats])
                        
                    if cl == '[var]':
                        for x in range(len(temp_searchFormats)):
                            vx = temp_searchFormats[x][-1].copy()
                            if len(vx) > 0:
                                vx = vx[0]
                            else:
                                continue
                            
                            ali.append((vx, temp_dataMap[x], temp_ans[x].replace(vx, '', 1).strip()))

##                    else:
##                        cl_db = {cl:1.0*class_infl}

                    self.show_process('variable keywords = {}'.format(ali))           
                
                    for ansx in ali:
                        rel = [x[0].replace(ansx[2], '', 1).strip() for x in self.sort_dict(self.getPartOfs(ansx[2])) if x[0].replace(ansx[2], '', 1).strip() != ansx[0] and x[0].replace(ansx[2], '', 1).strip() != '']
                        cm = [self.getCommonFormat(self.formatOutput(ansx[1], (z, [])), data) for y in [self.getCommon([[x[0] for x in self.sort_dict(self.getPartOfs(xx))], [x[0] for x in self.sort_dict(self.getPartOfs(ansx[0]))]]) for xx in rel] for z in y]
                        
                        cm_all = [x[0] for x in cm]
                        cm_set = set(cm_all)
                        n = 0
                        for x in cm_set:
                            if cm_all.count(x) > n:
                                n = cm_all.count(x)
                                val = x

                        cm = [x for x in cm if x[0] == val]
                        cmi = [ansx[-1] for x in cm]
                        
                    cmx = {}
                    cmxi = []
                    for i in range(len(cm)):
                        x = cm[i]
                        if x[0] in cmx:
                            cmx[x[0]].append(x[1][0])
                            cmxi.append(cmi[i])
                            
                        else:
                            cmx.setdefault(x[0], [x[1][0]])
                            
                cmx_infl = {x:len(cmx[x]) for x in cmx}
                pd = self.sort_dict(cmx_infl)
                if len(pd) > 0:
                    cmx_val = pd[0][0]
                else:
                    cmx_val = None
                self.show_process('possible data source = {}'.format(cmx))           

                if cmx_val != None:
                    varz = {}
                    print('xnew_form = ', cmx[cmx_val])
                    for cn in range(len(cmx[cmx_val])):
                        cx = cmx[cmx_val][cn]
                        varz.setdefault(cx, xtemp_qna_infl[cmi[cn]])
                        vcl = self.orClass(cx, sep=" ")
                        common_classes.append(vcl)

                    self.show_process("xvarz = {}".format(varz))
                    starters = self.generate_starters(varz)
                    self.show_process('xstarters = {}'.format(starters))
##--------------------------------------------------this is the computation of cases which reply is in memory b4 hand--------------------------------------------------
                temp_que = [que[y] for y in output_class if y not in false_list]
                
                temp_ans = [ans[y] for y in output_class if y not in false_list]

                temp_qna_infl = [qna_infl[y] for y in output_class if y not in false_list]
                
                #this maps the related que to data
                temp_dataMap = [dataMap[y] for y in output_class if y not in false_list]
                
                #this gets the common que in ans   
                temp_commonFormatQue = [commonFormatQue[y] for y in output_class if y not in false_list]
                    
                #this gets the common anns in que 
                temp_commonFormatAns = [commonFormatAns[y] for y in output_class if y not in false_list]

                #this uses the datamap to turn the individual que to look as that of data
                temp_searchFormats = [searchFormats[y] for y in output_class if y not in false_list]
                temp_searchFormat_max = self.getModelMax([x[0] for x in temp_searchFormats])
                
                varz = {}
                print('new_form = ', temp_commonFormatAns)
                for cn in range(len(temp_commonFormatAns)):
                    cx = temp_commonFormatAns[cn]
                    for ct in range(cx[0].count('[var]')):  
                        varz.setdefault(cx[-1][ct], temp_qna_infl[cn])
                        vcl = self.orClass(cx[-1][ct], sep=" ")
                        common_classes.append(vcl)

                self.show_process("varz = {}".format(varz))
                starters = self.generate_starters(varz)
                self.show_process('starters = {}'.format(starters))
        
                temp_genScore = self.getScore(common_classes, strict=False)
##                self.show_process("temp_score = {}".format(self.sort_dict(temp_genScore)[:10]))
                
                common_cls = [] # this all the classification of the answer for creating events through predicted_list
                if len(temp_genScore) > 0:
                    temp_genScore_li = self.sort_dict(temp_genScore)
                    common_cls = [x for x in temp_genScore_li if x[-1] == temp_genScore_li[0][-1]]

##                print('common_cls = ', common_cls[:10])

                cmps = []
                   
                #if its a single data with multiple replies        
                ret = self.isset(data, temp_que, temp_ans, cl, temp_searchFormat_max, common_cls)
                if cl == '[var]' and ret != None and sum(temp_qna_infl) > 1:
                    self.show_process("common que with var ans: {}".format([x[0]+":"+str(x[1]) for x in ret]))
                    for r in ret:
                        cmps.append(r)

                self.show_process("class list_score = {}".format(self.sort_dict(cl_db)[:10]))
                self.show_process('index = {}, score = {}, sum_total = {}, sum = {}, ratio = {}'.format(output_classes[cl], temp_qna_infl[:5], sum(temp_qna_infl), sum(qna_infl), sum(temp_qna_infl)/sum(qna_infl)))
                
                if (cl == data or "[var]" not in cl) and sum(temp_qna_infl) > 1:
                    cmps.append((cl, -1, data, temp_searchFormat_max, cl, common_cls))
                    
                elif len(cl_db) > 0:
                    for x in cl_db:
                        ret = self.matchAns(data, x, starters)
                        for y in ret:
                            val = (cl.replace('[var]', y[0]), y[-1], data, temp_searchFormat_max, x, common_cls)
                            if val not in cmps: cmps.append(val)
                            
                self.show_process("cmps = {}".format([x[0] for x in cmps[:10]]))
                self.show_process()
                pr = self.getClassScore2({x[0]:'' for x in cmps}, temp_genScore, data=False, is_same=False, sep= " ")
                
                if pr != []:
                    cmps = [x for x in cmps if x[0] in pr]

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
            self.show_process("\nfinal suggested ans = {}".format([x[0]+":"+str(x[1]) for x in predicted_list]))
            return self.Filter(predicted_list)
        
    def matchAns(self, data, x, starters=[]):
        rev_session = self.session.copy()
        rev_session.reverse()
        length = len(self.session)
        
        t = (len(x.split())**-1)/2
        cmp = self.getRelated(x, sep=" ", treshold=t, strict=True)

        vs = [xx for xx in cmp if self.percentage_similarity(data, xx, sep=" ") > 0.1]

        li = []
        backup_li = []
        
        self.show_process('matching ans for {}: {} related = {}, count = {} to {}'.format(data, x, vs[:5], len(cmp), len(vs)))
        for vx in vs:
            i = -1
            if vx in self.session:
                i = length - 1 - rev_session.index(vx)
            dm =  self.mapStrings(x, vx)
##                            -----------------this is to get the corresponding [var] value in data map---------------------
            rep = " "+" ".join([xy[-1].replace((' '+xy[0]+' ').replace(' [var] ', '', 1).strip(), '', 1).strip() for xy in dm if ' [var] ' in ' '+xy[0]+' ']) +" "
            if self.is_in(x.replace('[var]', '').strip(), vx) and rep not in " "+data+" " and len(rep.strip()) > 0 and rep in " "+vx+" ":
##                self.show_process("{} {} {}".format(" "+x.replace('[var]', '').strip()+" ",'---in----',  " "+vx+" "))
                rep = rep.strip()
##                self.show_process('vx = {}:{}'.format(vx, cmp[vx]))
            
                if len(starters) > 0:
                    for xx in starters:
                        if rep.startswith(xx):
                            li = [(rep, i)]
                            break
                        
                else:
                    li.append((rep, i))
        return li
    
    def Filter(self, predicted):
        if len(predicted) > 0:
            i = 0
            for x in predicted:
                if x[1] > i:
                    reply = x[0]
                    i = x[1]

                elif x[1] == -1:
                    reply = x[0]

            self.reply = True
            self.reply_value = reply
            self.setReply(self.string, reply)
            self.last_predicted_list = self.predicted_list
            self.predicted_list = []
            return reply
        
        else:
            self.reply = False
                    
    
    def save2memory(self, data, artificial=0):
        if data in self.memory:
            if artificial == 1:
                self.memory[data]["ifreq"] = str(int(self.memory[data]["ifreq"]) + 1)
            else:
                self.memory[data]["freq"] = str(int(self.memory[data]["freq"]) + 1)
                
        else:
            self.memory.setdefault(data, self.PROPERTIES)

            if artificial == 1:
                self.memory[data]["ifreq"] = "1"
                self.memory[data]["freq"] = "0"
            else:
                self.memory[data]["freq"] = "1"
                self.memory[data]["ifreq"] = "0"
        self.save()
		        
    def toogle_state(self, var):
        if var:
            var = False
        else:
            var = True

    def event_process(self, data=None, cls=None):
        if cls == None:
            cls = self.getAllClasses([data])
            cls = sorted(cls)
            
        for ev in self.events:
            self.show_process('\nsearching ... '+self.events[ev]["id"])
            
            key = self.events[ev]["key"]
            print('key', key)
            if len(key.strip()) > 0:
                if all([" "+k+" " in " "+data+" " for k in key.split()]):
                    go = True
                else:
                    go = False
            else:
                if self.getRelation(data, self.events[ev]["id"]) < 0.01:
                    go = False
                else:
                    go = True
            print('before go')
            if go:#self.events_id[x] in self.getRelated(data, sep=" ", treshold=0.01, strict=True):
                print('after go')
                if self.is_in(ev, str(cls), 10e-01):
                    self.show_process('matching events = {} for {}'.format(self.events[ev]["id"], data))
                    dm =  self.mapStrings(data, self.events[ev]["id"])
                    rep = self.formatOutput(dm,(self.events[ev]["format"],[]))
                    print(dm, self.events[ev]['format'], rep)

                    cmd = rep
                    if self.events[ev]['codebase'] == '1':
                        rep = self.try_process(rep)
                        
                    else:
                        print(data, self.events[ev]['format'])
                        rep = self.matchAns(data, self.events[ev]['format'])
                        print(rep)
                    expected_ans = (ev, data, rep)
                    self.set_expected_ans(expected_ans, self.new_expected_ans)
                    
                    confidence = self.events[ev]['confidence']
                    if confidence != '0/0':
                        confidence = self.try_process(confidence)
                    else:
                        confidence = 0

                    try:
                        confidence = float(confidence)
                    except Exception as e:
                        confidence = 0
                        
                    if confidence == None:
                        confidence = 0
                    print('cmd = {}, expected ans = {}, confidence = {}'.format(cmd, expected_ans[-1], confidence))
                    if expected_ans[-1] != None: print('cmd = {}, expected ans = {}, confidence = {}'.format(cmd, expected_ans[-1], confidence))
                    if expected_ans[-1] != None and dm != [] and confidence >= 0.5:
                        self.show_process('cmd = {}, expected ans = {}, confidence = {}'.format(cmd, expected_ans[-1], confidence))
                        self.used_event = ev
                        return expected_ans[-1]

                    
