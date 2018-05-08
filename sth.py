
    def fetchObjects(self, data):
        s = self.getSubsets(data)
        obj = {x:1 for x in s}
##        self.show_process('subsets = {}'.format(s))

        sub = {}
        for x in s:
            if x in self.memory:
                f = int(self.memory[x]["ifreq"])
                if f > 1:
                    f = f**-1
                sub.setdefault(x,f)
##        print(self.sort_dict(sub))
        for x in s:
            for y in x.split():
                if y in sub: obj[x] *= sub[y]
##        self.show_process('checking = {}, f = {}'.format(x, obj))
        pd = self.sort_dict(obj)
##        self.show_process('objs = {}'.format(pd))
        if len(pd) == 1:
            pass

        elif len(pd) > 0:
            m = pd[0][-1]
            sf = get_sf(m)
            if sf == 0:
                sf = 1

            m = (10**(-1*sf))*0.75
            objs = obj.copy()
            obj = [x[0] for x in pd if x[-1] >= m]
##            print(obj)
            exobj = [x[0] for x in pd if x[-1] < m]
            for x in exobj:
##                print('testing', x)
                if any([y in x for y in obj]):
                    obj.append(x)
                        
                else:
                    break
                    
##            print(obj)
            obj2 = []

            while obj != obj2:
                obj2 = obj.copy()
                
                for n in range(len(obj2)):
                    for x in obj2:
                        if x != obj2[n] and x in obj2[n]:
##                            print(x,'---', obj2[n])
                            if x in obj:
                                obj.remove(x)
                            break
##            print(obj)
        return obj
    
    def generateStarters(self, li):
        val = []
        length = sum([li[x] for x in li])
        if length < 1:
            return []
        
        c = {}
        val_n = {}
        for x in li:
            st = x.split()
            for n in range(1, len(st)+1, 1):
                v = " ".join(st[:n])
                val.append(v)
                if v in val_n:
                    val_n[v] += li[x]
                else:
                    val_n.setdefault(v, li[x])

                if v in c:
                    c[v].append(x)
                else:
                    c.setdefault(v, [x])
        val_s = set(val)
        val = [x for x in val_s if val.count(x)*val_n[x] > 1]
        n = 0
        chkd = []
        for x in val:
            for y in c[x]:
                if y not in chkd:
                    n += li[y]
                    chkd.append(y)
        self.show_process('predicted starters = {}, no = {}, sum total = {}, ratio = {}'.format(val, n, length, n/length))
        if n/length >= (1 - (length**-1)):
            return val
        else:
            return []
    
    def searchCodebase(self, data, que, ans):
        cdv = {}
        ac = []
        
        for a in ans:
            if a in self.rev_codebase and self.rev_codebase[a] in que:
                if self.rev_codebase[a] not in cdv:
                    cdv.setdefault(self.rev_codebase[a], a)
                    ac.append(self.getAllClasses([self.rev_codebase[a]]))
        ac = sorted(self.getCommon(ac))
        allcls = {}

        if len(cdv) > 0:
            for a in cdv:
                clses = self.common_classes(ac, data)
                allcls.setdefault(data, clses)

        return allcls

    def dist(self, data, x):
        c = 0
        st = data.split()
        xs = x.split()
        for vn in range(len(xs)):
            vx = xs[vn]
            if vx in st:
                c += (1 - abs((vn/len(xs)) - (st.index(vx)/len(st))))
                st[st.index(vx)] = "`"
        return c

    def zzz(self, data):
        li = {}
        for x in self.memory:
            c = self.dist(data, x)
            if c > 0:
                li.setdefault(x, c)
        pd = self.sort_dict(li)
        print(pd[:10])
        
    def findAnswer(self, data, ans):
        que = {}
        for x in self.memory:
##            if all([xx in x and data.split().count(xx) == x.split().count(xx) for xx in data.split()]):
            for y in self.memory[x]["ans"]:
                if y == ans and x != data:
                    c = self.dist(data, x)
                    if c > 0:
                        que.setdefault(x, c)
        pd = self.sort_dict(que)
        m = 0

        if len(pd) > 0:
            if pd[0][0] == data and len(pd) > 1: m = 1
            que = [x for x in que if que[x] >= pd[m][-1]]
            return que
        else:
            return []
    
    
    def setExpectedAns(self, expected_ans, general_expected_ans):
        if (len(self.expected_ans) == 0):
            general_expected_ans.append(expected_ans)
        else:
            if any([((x[1] == expected_ans[1]) and (x[-1] == expected_ans[-1])) for x in self.expected_ans]):
                pass
            else:
                general_expected_ans.append(expected_ans)
        
    def commonClasses(self, ac, val):
        rel = self.sort_dict(self.getRelated(val, sep=' ', treshold=0.1, strict=True, length=False, db=self.codebase))
        m = 1.0
        if len(rel) > 1 and rel[0][-1] == 1.0:
            m = rel[1][-1]
        rel = [x[0] for x in rel if x[-1] > 0 and x[-1] >= rel[0][-1]*m and self.is_in(str(ac), str(self.getAllClasses([x[0]])))]
        clses = self.getCommon([ac, self.getCommon([self.getAllClasses([x]) for x in rel])])
        
        return clses
        
    def isIn(self, a, b, limit=0.1):
        limit_val = ''
        ret = True
        
        length =  (len(a)*len(b))   
        if len(a) == 0 or len(b) == 0:
            a = b = ''
            ret  = False

        for x in a:
            if x not in b:
                limit_val += x
                if len(limit_val)/length > limit:
                    ret = False
                    break
            else:
                b = b.replace(x, "", 1)

##        if len(limit_val)/length > 0:
##            print('------------', len(limit_val)/length)
        return ret

    
    

    def processEvent(self, values, codebase, key):
        for val in values:
            self.set_event(val, values[val],  codebase, '0/0', val, key)

    def confirmEvent(self, data, suspected_que):
        if len(self.expected_ans) > 0:
            if any([(x[1] == suspected_que and x[-1] != None) for x in self.expected_ans]):
                for x in self.expected_ans:
                    e_a = False
                    if x[1] == suspected_que and x[-1] != None:
                        e_a = x
                        if x[-1] == data:
                            confidence =  (1,1)
                            self.confirmed_event = True
                        else:
                            confidence = (0, 1)
                        
                    if e_a !=False:
                        self.increase_confidence(e_a[0], confidence)
        self.show_process("confirming event {} => {} => {}".format(suspected_que, data, self.confirmed_event))
        return self.confirmed_event
    
    def setEvent(self, data, data_class,  codebase, confidence, val, key='', cls2=[]):
        if str(sorted(data_class)) not in self.events:
            self.show_process('creating event===============' + data)
            self.create_event(data, str(sorted(data_class)), codebase, '0/0', val, key, str(sorted(cls2)))
##            input('enter!')

    def increaseConfidence(self, event_name, value):
##        print('increasing confidence of '+event_name, value)
        confidence = self.read_event(event_name, 'confidence')
        a,b = confidence.split('/')
        a, b = int(a), int(b)
        a, b = a+value[0], b+value[1]
        self.update_event(event_name, "confidence", str(a)+'/'+str(b))
        self.show_process("affecting event confidence {} => {}".format(self.events[event_name]["id"], value))
        
    def createEvent(self, value, cls, codebase, confidence, Format, key, cls2=''):
        self.events.setdefault(cls, {"id":value, "classes":cls, "codebase":str(codebase), "confidence":confidence, "format":Format, "key":key, "classes2":cls2})
        self.save()

    def updateEvent(self, cls, key, value):
        self.events[cls][key] = value
        self.save()

    def readEvent(self, event_id, key):
        return self.events[event_id][key]
    
    def tryProcess(self, cmd):
        try:
            rcmd = eval(cmd)
            return str(rcmd)
        
        except Exception as e:
            return None
            
    def inheritProperties(self, data, related):
        ad_li = [] #list of possible classes
        m = min([len(x) for x in related])
        
        for n in range(m):
            cl = self.getClassIntersect([x[n] for x in related])
            if "[var]" in cl: cl.remove("[var]")
            print(cl)
            for c in cl:
                ad = c.replace("[var]", data)
                if ad not in ad_li and ad not in self.memory:
                    print("\ndata = {}\nad = {}\nrelated = {}\n".format(data, ad, related))
                    ad_li.append(ad)
        for x in ad_li: self.save2memory(x, 1)
        
    def generateArtificialData(self, data):
        related = self.bestRelated(data, multiply_factor=0.5, length=1, treshold=0.1, strict=True)
        related = [x for x in related if self.getClasses(x) != ["[var]"]]
##        if not self.learning: print(related)
        ad_li = []
        
        if len(related) > 1:
            cl = self.getClassIntersect(related)
            if "[var]" in cl: cl.remove("[var]")
           
            for c in cl:
                ad = c.replace("[var]", data)
                if ad not in ad_li and ad not in self.memory:
                    print("\ndata = {}\nad = {}\nrelated = {}\n".format(data, ad, related))
                    ad_li.append(ad)
        for x in ad_li: self.save2memory(x, 1)
                
    def bestRelated(self, data, multiply_factor=0.0, strict=True, treshold=0.0, length=False, sep="", db=None):
        rel = self.getRelated(data, sep, treshold, strict, len(data), db=db)
        if length != False:
            rel = {x:rel[x] for x in rel if len(x.split()) == length}
            
##        if not self.learning: print(data, self.sort_dict(rel))

        related = {}
        for x in rel:
            if sep == "" and len(x) == len(data):
                related.setdefault(x, rel[x])

            if sep == " " and len(x.split()) == len(data.split()):
                related.setdefault(x, rel[x])
                
        if strict and data in related:
            related.pop(data)
            
        pd = self.sort_dict(related)
##        if not self.learning: print(data, pd)
        rel = [x[0] for x in pd if x[-1] > 0 and x[-1] >= pd[0][-1]*multiply_factor]
        return rel

    
    def andClass(self, data, sep=""):
        if sep == "":
            dl = list(data)
        else:
            dl = data.split(sep)
        classes = []
        for x in dl:
            classes.append(self.getClasses(x))

        common = self.getCommon(classes)
        common = sorted(common)
        return common

    def orClass(self, data, sep=""):
        if sep == "":
            dl = list(data)
        else:
            dl = data.split(sep)
        allclasses = []
        for x in dl:
            allclasses.extend(self.getAllClasses(dl))
        return allclasses
    
    def factorize(self,string1,string2):
        subset1 = self.getSubsets(string1)
        intersect = ""
        string2 = " "+string2.strip()+" "
        string1 = " "+string1.strip()+" "
        for x in subset1:
            if " "+x+" " in string2 and ((string2.index(" "+x+" ") == 0) or (string2.index(" "+x+" ")+len(" "+x+" ") == len(string2))) and ((string1.index(" "+x+" ") == 0) or (string1.index(" "+x+" ")+len(" "+x+" ") == len(string1))):
                if len(x) > len(intersect):
                    intersect = " "+x+" "
        if len(intersect) > 0:
            factor1 = string1.replace(intersect,"",1).strip()
            factor2 = string2.replace(intersect,"",1).strip()
            intersect = intersect.strip()
            return factor1,factor2,intersect
        else:
            return "","",""
    
    
   

    

    ##rates the score of predicted vals rel to ref scores
    def getClassScore2(self, predicted, gen_score, data=False, is_same=False, sep= " "):
        ansval_infl = {x:0 for x in predicted}

        for y in predicted:
            allclasses = []
            if data == False:
##                print(y)
                myclasses = self.orClass(y, sep=" ")
##                print('[var] is a place' in myclasses)
##                print('[var] is a place' in gen_score)
                myclasses = self.getScore([myclasses])   
##                print(self.sort_dict(myclasses)[:5])
##                print()
                for c in myclasses:
                    if c in gen_score:
                        ansval_infl[y] += gen_score[c]
                        
        if len(ansval_infl)> 0 and all([[a for a in ansval_infl.values()][0] == ansval_infl[x] for x in ansval_infl]):
            return []
        else:
            ansval_infl = [x for x in ansval_infl if ansval_infl[x] > 0]
            return ansval_infl
    
    ##rates the score of predicted vals rel to ref scores
    def getClassScore(self, ansval, score, data=False, is_same=False, sep= " "):
        ansval_infl = {x:0 for x in ansval}

        for y in ansval:
            allclasses = []
            if data == False:
                myclasses = self.andClass(y)
                
                if sorted(score) == sorted(myclasses):
                    ansval_infl[y] += 1.0
            else:
                xx = self.getCommonFormat(y, data)
                if len(xx[-1]) > 0 and  is_same == False:
                    if sep == "":
                        dl = []
                        for x in xx[-1]:
                            dl.extend(self.getAllClasses(list(x)))
                            
                    else: dl = self.getAllClasses(xx[-1])
                    allclasses.extend(dl)
                    
                else:
                    if sep == "": dl = list(xx[0].replace("[var]",""))
                    else: dl = [xx[0].replace("[var]","")]
                    allclasses.extend(self.getAllClasses(dl))

                myclasses = self.getScore([allclasses])   

                for c in score:
                    if c in myclasses:
                        ansval_infl[y] += score[c]
        return ansval_infl

    def getClassIntersect(self, li, strict=True):
        c = -1
        intersect = []
        for k in li:
            c += 1
            if type(k) == list:
                allclasses = self.getAllClasses(k)
            else:
                allclasses = self.getAllClasses([k])

            if strict ==False and allclasses == []:
                pass
            else:
                if c == 0:
                    intersect = [x for x in self.getScore([allclasses])]
                else:
                    intersect = self.getCommon([intersect, [x for x in self.getScore([allclasses])]])
            
        return intersect 

    def sortLast(self, li):
        i = []
        sli = self.session.copy()
        sli.reverse()
        
        for x in li:
            if x in sli:
                i.append(sli.index(x))
            else:
                i.append(-1)
        if i == []:
            return -1
        else:
            m = i.index(min(i))
            while all([-1 == x for x in i]) == False and m == -1:
                i[i.index(min(i))] = max(i)
                m = i.index(min(i))
                
            return m
    
    
        
    def getScore(self, classes, strict=True):
        d = {}
        l = [len(x) for x in classes]
        if strict:
            Type = 'all'
        else:
            Type = 'any'
            
        if len(l) > 0:
            i = l.index(max(l))
            for x in classes[i]:
                if self.in_list(x, classes, Type):
                    if x in d:
                        d[x] += 1
                    else:
                        d.setdefault(x,1)
            tot = sum([d[x] for x in d])
            d = {x:d[x]/tot for x in d}
            return d
        else:
            return {}

    def inList(self, data, li, Type):
        if Type == 'any':
            return any([data in a for a in li])
        else:
            return all([data in a for a in li])

    
    def tryRead(self, filename):
        try:
            return self.read(filename)
        except Exception as e:
            return []
        
    def saveData(self, filename, data):
        data = [x + "\n" for x in data]
        self.write(filename, data)

    def saveInput(self, objectname):
        if objectname not in self.memory:
            #save input to  memory
            for x in objectname.split():
                self.save2memory(x)
            self.save2memory(objectname)
        
    def setReply(self, objectname, reply, silent=False):
        if silent == False:
            self.setContext(reply)
        if reply not in self.memory:
            #save input to  memory
            self.save2memory(reply)
        if objectname not in self.memory:
            self.save2memory(objectname, 1)
        data = self.memory[objectname]["ans"]

        if reply in data:
            self.memory[objectname]["ans"][reply] = str(int(data[reply]) + 1)
            
        else:
            self.memory[objectname]["ans"].setdefault(reply, "1")

    def load(self, filename):
        var = [x.replace("\n", "") for x in self.read(filename)]
        return var

    def loadContext(self):
        self.context = self.load("context")

        
    def loadEvents(self):
        self.events = self.read_json("events")
        
    


    
    
    

    
        
    def getAllClasses(self, li):
        allclasses = []
        for x in li:
##            for y in x.split(" "):
##                l = self.getClasses(y)
##                for a in l:
##                    if a not in allclasses:
##                        allclasses.append(a)
            l = self.getClasses(x)
            for b in l:
                if b not in allclasses:
                    allclasses.append(b)
            
        return allclasses

    def getMRelated(self, li):
        ref  = {}
        for x in li:
            rel = self.getRelated(x.replace("[var]","").strip())

            for v in rel:
                if v in ref:
                    ref[v] += rel[v]
                else:
                    ref.setdefault(v, rel[v])
        return ref

    def intersect(self, dict1, dict2):
        l = [dict1, dict2]
        li = [len(x) for x in l]
        ref = l[li.index(min(li))]
        l.pop(li.index(min(li)))
        other = l[0].copy()
        
        new_dict = {}
        for x in ref:
            if x in other:
                new_dict.setdefault(x, ref[x]*other[x])
        return new_dict
    
    def readFreq(self, data):
        return self.memory[data]["freq"]

    def getQueAns(self, datalist):
        que = []
        ans = []
        infl = []
        for x in datalist:
            for y in self.memory[x]["ans"]:
                if len(y) > 0:
                    ans.append(y)
                    que.append(x)
                    infl.append(int(self.memory[x]["ans"][y]))
        return que, ans, infl

    
