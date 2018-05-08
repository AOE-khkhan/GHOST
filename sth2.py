
    
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
    
    def preProcess(self, data, data_after):
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

    
