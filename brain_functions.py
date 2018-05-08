import os, json, time

from functions import *

class BrainFunctions:
    def createFile(self, name, path="memory\\console\\"):
        file = open(path+name,"w")
        file.write("")
        file.close()
        
    def createMemory(self):
        if not os.path.exists('memory/console/memory_index.json'):
            self.setup()
        for x in ['memory_index.json', 'session_index.json', 'context.json', 'blocks/0.json', 'ref/0.json', 'sessions/0.json', 'memory_log.json']:
            if x.endswith('.json'):
                d = {}
                if 'memory_index' in x:
                    d = {0:0}
                elif 'memory_log' in x:
                    d = {'last_index':-1, 'last_block':0, 'last_ref':0, 'last_session':0, 'last_session_index':-1, 'last_ref_index':-1}
                elif 'session_index' in x:
                    d = {0:0}
                self.writeJson(x[:-5], d)
            else:
                self.createFile(x)
        self.createFile('text.txt', 'resources\\')

    def createSession(self, data, session_index):        
        session = {}
        ind = str(self.resetMemoryLogId('last_session_index'))
        session.setdefault(ind, {'0':data})

        self.writeSession(str(session_index), session)
        data_sessions = self.locateMemoryData(data, "sessions")
        data_session = ind+" "+str(0)

        if session_index in data_sessions and data_session not in data_sessions[session_index]:
            data_sessions[session_index].append(data_session)

        else:
            data_sessions.setdefault(session_index, [data_session])
        self.setLocatedMemoryData(data, 'sessions', data_sessions)

    def createRef(self, data, index, ref_index):
        memory_index = self.readJson('memory_index')
        memory_index.setdefault(ref_index, ref_index)
        self.writeJson('memory_index', memory_index)
        self.writeRef(str(ref_index), {index: data})
        

    def dataInMemory(self, data):
        memory_index = self.readJson('memory_index')
        for index in memory_index:
            indexes = self.readJson('ref/'+index)
            indxs = []
            for i in range(len(indexes)):
                indxs.append(indexes[str(i)])

            if data in indxs:
                val = str(index)+"."+str(indxs.index(data))
                return val
        else:
            return False

    def formatOutput(self,dataMap,ansFormat):
        output = ""
        out = " "+str(ansFormat[0])+" "
        ans = " ".join([x[-1] for x in dataMap])

        if ans == ansFormat[0]:
            for x in dataMap:
                output += x[0]+ " "
        else:
            for d in dataMap:
                out = out.replace(" "+d[-1]+" ", " "+d[0]+" ", 1)
            output = out.strip()

        return output.strip()
    
    def generateModelers(self, data, que, ans):
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

        return dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats
           
    def genNextId(self, key):
        sd = self.readJson('memory_log')
            
        sd[key] += 1
        index = sd[key]
        
        self.writeJson('memory_log', sd)
        return index

    def getCommonFormat(self,que,ans):
        q = que.split(" ")  #list(que)
        a = ans.split(" ")  #list(ans)
        f = ""
        r = []
        for x in q:
            if x in a:
                f += " "+x
                a[a.index(x)] = "`"
            else:
                
                if f.endswith(" [var]") == False:
                    f += " [var]"
                    r.append(x)
                else:
                    r[-1] += " "+x
                
        return (f.strip(),r)
    
    def getClasses(self, data):
        li = []
        for x in self.memory:
            val = " "+x+" "
            d = " "+data+" "
            b = False
            while d in val:
                b = True
                val = val.replace(d, " [var] ", 1)
            if b:
                if val.strip() not in li:
                    li.append(val.strip())

        return li

    def getCommon(self, li):
        l = [len(x) for x in li]
        if l == []:
            return []
        else:
            mi = l.index(min(l))
            common = []
            for x in li[mi]:
                if all([x in cl for cl in li]) and x not in common:
                    common.append(x)
            return common
    
    def getDataIndexFromSession(self, index):
        session_index, session_key, sid = index.split(".")
        return self.readSession(str(session_index))[session_key][sid]
   
    def getFeatures(self, data, callback=False):
        self.showProcess("data = {}".format(data))
        if callback == True:
            que, ans, qna_infl, que_sessions, ans_sessions, in_memory = self.xxque, self.xxans, self.xxqna_infl, self.xxque_sessions, self.xxans_sessions, self.xxin_memory

        else:
            que, ans, qna_infl, que_sessions, ans_sessions, in_memory = self.getQnA(data)

        dataMap = commonFormatQue = commonFormatAns = searchFormats = outputFormats = output_classes = []
##-----------------------------------------this is for inheritance of properties when no ans---------------------------------------------------------------------
                    
        if len(ans) > 0:
            dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats = self.generateModelers(data, que, ans)

            #keep the originals
            original_ans = ans.copy()
            original_commonFormatAns = commonFormatAns.copy()
            original_outputFormats = outputFormats.copy()

            # for i in range(len(que)):
            #     self.showProcess("\nque = {}\nans = {}\ndatamap = {}\ncommonFormatQue = {}\ncommonFormatAns = {}\noutputFormat = {}\nsearchFormat = {}\nque_sessions = {}\nans_sessions = {}\nin_memory = {}\n".format(
            #         que[i], ans[i], dataMap[i], commonFormatQue[i], commonFormatAns[i], outputFormats[i], searchFormats[i], que_sessions[i], ans_sessions[i], in_memory[i]))

            outputFormats_infl = {x:outputFormats.count(x)/len(outputFormats) for x in set(outputFormats)}
            pd = self.sortDict(outputFormats_infl)
            infl = {x:outputFormats_infl[x] for x in outputFormats_infl if outputFormats_infl[x] > 0}

##------------------------this classifies the output formats and studies each to get reply---------------------------------------------------------------------
            output_classes = {x.strip():[i for i in range(len(outputFormats)) if x == outputFormats[i]] for x in set(outputFormats)}
            
            self.showProcess("output_classes = {}\n".format(output_classes))
        return que, ans, qna_infl, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, output_classes, que_sessions, ans_sessions, in_memory

    def getMax(self, Model):
        d = {}
        model = ''
        for n in range(len(Model)):
            if Model[n] in d.keys():
                d[Model[n]] += 1
            else:
                d.setdefault(Model[n], 1)
                
        ind = [x for x in d.values()]
        
        if len(ind) > 0:
            m = max(ind)

            if ind.count(m) == 1:
                index = ind.index(m)
                model = [x for x in d.keys()][index]
        return model
    
    def getMemoryData(self, data, key):
        index = self.dataInMemory(data)
        if index == False and type(index) == bool:
            pass
        
        else:
            return self.locateMemoryData(index, key)

    def getQueAns(self, datalist, rel_index=None):
        que = []
        ans = []
        infl = []
        que_sessions = []
        ans_sessions = []
        in_memory = []
        for x in datalist:
            if rel_index == None:
                sess = self.getMemoryData(x, "sessions")

            else:
                sess = self.locateMemoryData(rel_index[x], "sessions")
            
            for sess_id in sess:
                sf = self.readSession(sess_id)

                for si in sess[sess_id]:

                    #'i' is the index of the data in sess list
                    sess_index, data_sess_index = str(si).split(" ")
                    i = int(data_sess_index)
                    li = sf[sess_index]
                    
                    if i+1 <= len(li)-1:
                        y = li[str(i+1)]
                        ai = sess_id+"."+sess_index+"."+str(i+1)

                    elif i == len(li)-1:
                        if int(sess_index)+1 > self.BLOCK_SIZE:
                            y = self.readSession(str(int(sess_id)+1))["0"]['0']
                            ai = str(int(sess_id)+1)+".0.0"

                        if str(int(sess_index)+1) in sf:
                            y = sf[str(int(sess_index)+1)]['0']
                            ai = sess_id+"."+str(int(sess_index)+1)+".0"

                        else:
                            continue
                    
                    else:
                        continue

                    ans.append(self.locateMemoryData(y, "text"))
                    que.append(x)

                    qsi = sess_id+"."+sess_index+"."+str(i)
                    que_sessions.append(qsi)
                    ans_sessions.append(ai)
                    
                    # print("sess id={}, si = {}, sf = {}".format(sess_id, si, sf))
                    # print(ans[-1], x, qsi, ai)
                    # input()
                    
                    ans_sess = self.locateMemoryData(y, "sessions")
                    m = None
                    for sfi in ans_sess:
                        for sfx in ans_sess[sfi]:
                            ssi, sdi = sfx.split(" ")
                            asi = sfi+"."+ssi+"."+sdi
                            asi_value = int(asi.replace(".", ""))

                            if m == None:
                                m = asi_value

                            if m != None and asi_value < m:
                                m = asi_value
                    
                    qsi_value = int(qsi.replace(".", ""))
                    ai_value = int(ai.replace(".", ""))

                    if m < qsi_value:
                        im = True

                    else:
                        im = False
                    in_memory.append(im)

                    # li.remove(rel_index[x])
                    infl.append(1)

        return que, ans, infl, que_sessions, ans_sessions, in_memory

    def getQnA(self, data):
        related, rel_index = self.getQueRelated(data)
        relatedlist = [x for x in related]
        self.showProcess('related = {}, count = {}'.format(relatedlist[:10], len(relatedlist)))    

        que, ans, qna_infl, que_sessions, ans_sessions, in_memory = self.getQueAns(relatedlist, rel_index)
        
        return que, ans, qna_infl, que_sessions, ans_sessions, in_memory

    def getQueRelated(self, data):
        if len(data.split()) > 0:
            rel, rel_index = self.getRelated(data, " ", (len(data.split())**-1)/2, strict=True)
            related = rel.copy()
            for x in rel:
                if len(x.split()) != len(data.split()):
                    related[x] *= 0.5
            pd = self.sortDict(related)

            m = 0
            if len(pd) > 0:
                if pd[0][0] == data and len(pd) > 1: m = 1

                related = {x:related[x] for x in related if related[x] >= pd[m][-1]*0.5}
                return related, rel_index
            
            else:
                return {}, {}
        else:
            return {}, {}

    def getRelated(self, data, sep="", treshold = 0.2, strict=False, length=False):
        result = {}
        rel_index = {}
        memory_index = self.readJson('memory_index')
        for mi in memory_index:
            memory_ref = self.readRef(str(mi))
            for valx in memory_ref:
                val = memory_ref[valx]
                rel = self.getRelation(data, val, sep)
                if rel >= treshold:
                    result.setdefault(val, rel)
                    rel_index.setdefault(val, mi+"."+valx)
                
        return result, rel_index

    def getRelation(self, data, string, sep=""):
        infl1 = self.percentageSimilarity(data, string, sep)
        infl2 = self.percentageSimilarity(string, data, sep)
        return formatVal((infl1*infl2))

    def getRelationIntersect(self, que_val, ans_val):
        que_rel, que_rel_index = self.getRelated(que_val, " ", (len(que_val.split())**-2)/2, strict=False)

        #check ans in memory
        ans_rel, ans_rel_index = self.getRelated(ans_val, " ", (len(ans_val.split())**-2)/2, strict=False)

        #check intersection of que and ans in memory
        if len(ans_rel_index) <= len(que_rel_index):
            val_li = ans_rel
            val_li2 = que_rel

        else:
            val_li = que_rel
            val_li2 = ans_rel

        rel_index = {x:val_li[x]*val_li2[x] for x in val_li if x in val_li2}
        return rel_index

    def getPartOfs(self, data):
        li = []
        que = []
        ans = []
        infl = []
        for ind in self.memory_index:
            memory_ref_data = self.readRef(ind)
            
            for x in memory_ref_data:
                if data in x and x != data:
                    li.append(x)
        return {x:self.getRelation(data, x, sep=" ") for x in li}      

    def info(self, string): #to show information
        if self.show_info:
            print(string)
        
    def loadMemory(self):
        if os.path.exists('memory/console/memory_index.json'):
            d = self.readJson("context")
            self.context = []
            for i in range(100):
                if str(i) in d:
                    self.context.append(d[str(i)])
                else:
                    break
            self.memory_index = self.readJson("memory_index")

        else:
            self.setup()
            self.createMemory()
    
    def locateMemoryData(self, index, key):
        block, index = index.split(".")
        if key == "text":
            ref = self.readJson('ref/'+block)
            return ref[index]

        else:    
            memory_data = self.readJson('blocks/'+block)
            if index in memory_data:
                value = memory_data[index]
                
                if key == 'sessions':
                    if "sessions" in value:
                        value = json.loads(value[key])
                    else:
                        value = {}
                else:
                    value = value[key]
                return value

    def log(self, data, data_index):
        if self.lastInputSource != self.source: self.genNextId("last_session_index")
        
        self.setContext(data)
        self.setSession(data_index)
        self.lastInputSource = self.source

    def mapStrings(self, str1, str2):
        s1 = str1.split(" ")#string 1
        s2 = str2.split(" ")#string 2

        l = []
        lmap = []
        n = -1
        n2 = -1
        s = s1.copy()

        order = [] #order, its to organize the similar value index such that they are subst orderly
        s1t, s2t = s1.copy(), s2.copy()
        for x in s1t:
            if x in s2t:
                order.append(s2t.index(x))
                s2t[s2t.index(x)] = 'null'

        for x in range(len(order)):
            if x == 0:
                continue
            else:
                if order[x] < order[x-1]:
                    order[x] = -1
        order = [x for x in order if x != -1]

        if len(s1) == len(s2):
            l = [s1[x] for x in range(len(s1))]
            lmap = [s2[x] for x in range(len(s1))]

        else:
            for x in range(len(s)):
                if s[x] in s2 and s2.index(s[x]) in order:
##                    print(s1, '===', s2, s[x])
                    if (abs(n-s1.index(s[x])) > 0 and len(" ".join(s1[n+1:s1.index(s[x])]).replace("`","").strip()) > 0) or x == len(s2)-1:
##                        print('a')
                        if s1.index(s[x]) == len(s1)-1 or x == len(s2)-1:
                            l.append(" ".join(s1[n+1:]).strip())
                            lmap.append(" ".join(s2[n2+1:]).strip())
                            
                        else:
##                            print('c')
                            l.append(" ".join(s1[n+1:s1.index(s[x])+1]).strip())
                            lmap.append(" ".join(s2[n2+1:s2.index(s[x])+1]).strip())
                            
                        for vn in range(n2+1,s2.index(s[x]),1):
                            s2[vn] = "`"
                        for vn in range(n+1,s1.index(s[x]),1):
                            s1[vn] = "`"
                    else:
##                        print('b')
                        if s1.index(s[x]) == len(s1)-1:
                            l.append(" ".join(s1[n+1:]).strip())
                            lmap.append(" ".join(s2[n2+1:]).strip())
                            
                        else:
##                            print('d')
                            l.append(" ".join(s1[n+1:s1.index(s[x])+1]).strip())
                            lmap.append(" ".join(s2[n2+1:s2.index(s[x])+1]).strip())
                            
                            
                    n = s1.index(s[x])
                    n2 = s2.index(s[x])
                    s2[s2.index(s[x])] = "`"
                    s1[s1.index(s[x])] = "`"
                else:
                    if s1.index(s[x]) == len(s1)-1:
                        l.append(" ".join(s1[n+1:]).strip())
                        lmap.append(" ".join(s2[n2+1:]).strip())
        m = []
        for n in range(len(l)):
            m.append((l[n],lmap[n]))
        
        return m
    
    def memory(self):
        for data in range(self.readMemoryLog('last_ref')):
            yield data

    def percentageSimilarity(self, data, string, sep="", strict=True):
        if len(sep) < 1:
            d = list(data)
            s = list(string)
            
        else:
            d = data.split(sep)
            s = string.split(sep)
            
        c = 0
        for vn in range(len(d)):
            v = d[vn]
            inf = (len(d) - vn)/len(d)
            if v in s:
                if strict:
                    if d.index(v) == s.index(v):
                        c += 1.0*inf
                    else:
                        c += 0.5*inf
                else:
                    c += 1.0*inf
                s[s.index(v)] = "`"
                d[d.index(v)] = "`"
            else:
                d[d.index(v)] = "`"
        if len(s) > 0:
            infl1 = c/len(s)
        else:
            infl1 = 0
        return infl1
    
    def readBlock(self, block):
        return self.readJson('blocks/'+block)

    def readRef(self, ref):
        return self.readJson('ref/'+ref)

    def readMemoryLog(self, key=False):
        memory_log = self.readJson('memory_log')
        if False:
            return memory_log

        else:
            return memory_log[key]
        
    def readJson(self, name):
        file = open(self.CONSOLE_MEMORY_PATH+name+'.json', 'r')
        content = file.read()
        file.close()
        return json.loads(content)

    def readSession(self, session):
        return self.readJson('sessions/'+session)

    def resetMemoryLogId(self, key):
        sd = self.readJson('memory_log')
            
        sd[key] = 0
        index = sd[key]
        
        self.writeJson('memory_log', sd)
        return index
    
    def setContext(self, data):
        if len(self.context) == 100:
            self.context.pop(0)
        self.context.append(data)
        self.writeJson('context', {i:self.context[i] for i in range(len(self.context))})

    def setLocatedMemoryData(self, index, key, value):
        ret = index
        
        block, index = index.split(".")
        if type(value) == dict:
            value = json.dumps(value)
            
        if os.path.exists('memory/console/blocks/'+block+'.json'):
            memory_block_data = self.readBlock(block)

        else:
            memory_block_data = {}

        if index not in memory_block_data:
            memory_block_data.setdefault(index, {})

        if key in memory_block_data[index]:
            memory_block_data[index][key] = value
        else:
            memory_block_data[index].setdefault(key, value)

        self.writeJson('blocks/'+block, memory_block_data)
        return ret

    def setMemoryData(self, data, key, value):
        index = self.dataInMemory(data)
        if index == False and type(index) == bool:
            index = self.genNextId('last_index')
            last_ref = self.readMemoryLog('last_ref')
            
            if index < self.BLOCK_SIZE:
                self.updateRef(data, index, last_ref)
                
            else:
                index = self.resetMemoryLogId("last_index")
                self.createRef(data, index, self.genNextId('last_ref'))

            index = str(self.readMemoryLog('last_ref'))+"."+str(self.readMemoryLog('last_index'))
        return self.setLocatedMemoryData(index, key, value)

    def setSession(self, data_index):
        last_session = self.readMemoryLog('last_session')
        last_session_index = self.readMemoryLog('last_session_index')
        if last_session_index < self.BLOCK_SIZE:
            self.updateSession(data_index, last_session)
            
        else:
            self.createSession(data_index, self.genNextId('last_session'))

    def setup(self, li=None):
        if li == None:
            li = ['resources', 'memory', 'sessions', 'memory/console/', 'memory/console/blocks', 'memory/console/ref', 'memory/console/sessions']
        for x in li:
            if os.path.exists(x):
                pass
            else:
                self.info('setting up "{}"'.format(x))
                os.mkdir(x)

    def sortDict(self, dic, ascending=True):
        new = []
        if len(dic) > 0:
            d = dic.copy()
            while len(d) > 0:
                v = max(zip(d.values(), d.keys()))
                new.append((v[1],v[0]))
                d.pop(v[1])
        return new
    
    def showProcess(self, value=""):
        if self.show_process_state: print(value)

    
    def switchSource(self):
        self.lastSource = self.source
        if self.source == "x":
            self.source = "y"

        else:
            self.source = "x"

    def updateSession(self, data, session_index):
        session = self.readSession(str(session_index))
        ind = int(self.readMemoryLog('last_session_index'))
        
        if str(ind) in session:
            session[str(ind)].setdefault(str(len(session[str(ind)])), data)

        else:
            # print("not exist ", ind)
            session.setdefault(str(ind), {'0':data})

        self.writeSession(str(session_index), session)
        data_sessions = self.locateMemoryData(data, "sessions")
        data_session = str(ind)+" "+str(len(session[str(ind)])-1)

        # print("zzzzzzzzzz", data, data_sessions, data_session)
        if str(session_index) in data_sessions:
            data_sessions[str(session_index)].append(data_session)
        
        else:
            data_sessions.setdefault(session_index, [data_session])
        self.setLocatedMemoryData(data, 'sessions', data_sessions)

    def updateRef(self, data, index, ref_index):
        memory_ref = self.readRef(str(ref_index))
        memory_ref.setdefault(index, data)
        self.writeRef(str(ref_index), memory_ref)

    def writeRef(self, name, data):
        self.writeJson('ref/'+name, data)

    def writeSession(self, name, data):
        self.writeJson('sessions/'+name, data)
        
    def writeJson(self, name, dictionary):
        content = json.dumps(dictionary)
        file = open(self.CONSOLE_MEMORY_PATH+name+'.json', 'w')
        file.write(content)
        file.close()

    def writeRef(self, name, data):
        self.writeJson('ref/'+name, data)
