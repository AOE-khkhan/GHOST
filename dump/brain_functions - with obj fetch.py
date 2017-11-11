import os

from functions import *

class brain_functions:
    def fetch_object(self, data):
        s = self.getSubsets(data)
        obj = {x:1 for x in s}
##        self.show_process('subsets = {}'.format(s))
        for x in s:
            f = 0
            for y in self.memory:
                if x in y:
                    f += int(self.freq[self.memory.index(y)])
            self.show_process('checking = {}, f = {}'.format(x, f))
            if f > 1:
                obj[x] *= (f**-1)
        pd = self.sort_dict(obj)
        self.show_process('objs = {}'.format(pd))
        
        if len(pd) == 1:
            pass
        
        elif len(pd) > 0:
            m = pd[0][-1]*0.75
            li = [x[0] for x in pd if x[-1] >= m]
            exli = [x[0] for x in pd if x[-1] < m]

            self.show_process('factors = {}'.format(li))
            self.show_process('eliminators = {}'.format(exli))
        
            exli2 = exli.copy()
            
            for n in range(len(exli2)):
                for x in exli2:
                    if x != exli2[n] and x in exli2[n]:
                        print(x,'---', exli2[n])
                        if x in exli:
                            exli.remove(x)
                        break
            print(exli)
            newli = [x for x in li]
            for n in range(len(li)):
                for x in exli:
                    print(x,'---', newli[n])
                    if x in newli[n]:
                        newli[n] = newli[n].replace(x, '', 1)
                        break
            print(newli)

            obj = [x.strip() for x in set(newli) if x.strip() not in exli2 and x in data]
                
        return obj
    
    def generate_starters(self, li):
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
    
    def search_codebase(self, data, que, ans):
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
        
    def find_answer(self, data, ans):
        que = {}
        for x in self.memory:
##            if all([xx in x and data.split().count(xx) == x.split().count(xx) for xx in data.split()]):
            for y in self.read_data(x):
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
    
    def create(self, name, path="memory\\console\\"):
        file = open(path+name+".txt","w")
        file.write("")
        file.close()
        
    def createMemory(self):
        if not os.path.exists('memory/console/memory.txt'):
            self.setup()
        for x in ['memory', 'codebase', 'sessions', 'context', 'freq']:
            self.create(x)
        self.create('text.txt', 'resources\\')
        
    def set_expected_ans(self, expected_ans, general_expected_ans):
        if (len(self.expected_ans) == 0):
            general_expected_ans.append(expected_ans)
        else:
            if any([((x[1] == expected_ans[1]) and (x[-1] == expected_ans[-1])) for x in self.expected_ans]):
                pass
            else:
                general_expected_ans.append(expected_ans)
        
    def common_classes(self, ac, val):
        rel = self.sort_dict(self.getRelated(val, sep=' ', treshold=0.1, strict=True, length=False, db=self.codebase))
        m = 1.0
        if len(rel) > 1 and rel[0][-1] == 1.0:
            m = rel[1][-1]
        rel = [x[0] for x in rel if x[-1] > 0 and x[-1] >= rel[0][-1]*m and self.is_in(str(ac), str(self.getAllClasses([x[0]])))]
        clses = self.getCommon([ac, self.getCommon([self.getAllClasses([x]) for x in rel])])
        
        return clses
        
    def is_in(self, a, b, limit=0.1):
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

    def getModelMax(self, Model):
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
    
    def process_event(self, values, codebase, key):
        for val in values:
            self.set_event(val, values[val],  codebase, '0/0', val, key)

    def confirm_event(self, data, suspected_que):
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
    
    def set_event(self, data, data_class,  codebase, confidence, val, key='', cls2=[]):
##        print('for data {}'.format(data), end='')
##        if str(sorted(data_class)) in self.events:
##            print(', it is in events',[self.events_id[x] for x in self.events[str(sorted(data_class))]], end='')
##        else:
##            print(', its not in events', end='')
##        if str(sorted(data_class)) in self.events and any([self.events_id[x] in self.getRelated(data, " ", True) for x in self.events[str(sorted(data_class))]]):
##            print(' and in related')
##        else:
##            print(' not in related')
        data_rel = self.getRelated(data, " ", True)
        if str(sorted(data_class)) in self.events and any([self.events_id[x] in data_rel for x in self.events[str(sorted(data_class))]]):
            pass
        
        else:
            self.show_process('creating event===============' + data)
            self.create_event(data, str(sorted(data_class)), codebase, '0/0', val, key, str(sorted(cls2)))
##            input('enter!')

    def increase_confidence(self, event_name, value):
##        print('increasing confidence of '+event_name, value)
        confidence = self.read_event(event_name, 'confidence')
        a,b = confidence.split('/')
        a, b = int(a), int(b)
        a, b = a+value[0], b+value[1]
        self.update_event(event_name, self.read_event(event_name, 'classes'), self.read_event(event_name, 'codebase'), str(a)+'/'+str(b), self.read_event(event_name, 'format'), self.read_event(event_name, 'key'), self.read_event(event_name, 'classes2'))
        self.show_process("affecting event confidence {} => {}".format(self.events_id[event_name], value))
        
    def create_event(self, value, cls, codebase, confidence, Format, key, cls2=''):
        name = str(len(self.events_id))
        file = open('memory/console/events/'+name+'.txt', 'w')
        file.writelines([value+'\n', cls+'\n', str(codebase)+'\n', confidence+'\n', Format+'\n', key+'\n', cls2+'\n'])
        file.close()
        
        self.events_id.setdefault(name, value)
        self.saveData('events', self.events_id)
        if str(cls) in self.events:
            self.events[str(cls)].append(name)
            
        else:
            self.events.setdefault(str(cls), [name])
        self.save()

    def update_event(self, event_id, cls, codebase, confidence, Format, key, cls2=''):
        value = self.events_id[event_id]
        file = open('memory/console/events/'+event_id+'.txt', 'w')
        file.writelines([value+'\n', cls+'\n', str(codebase)+'\n', confidence+'\n', Format+'\n', key+'\n', cls2+'\n'])
        file.close()
        self.save()
        
    def read_event(self, event_id, key):
        key_db = {'id':0, 'classes':1, 'codebase':2, 'confidence':3, 'format':4, 'key':5, 'classes2':6}
        file = open('memory/console/events/'+event_id+'.txt')
        r = [x.replace('\n',"") for x in file.readlines()]
        file.close()
        return r[key_db[key]]
    
    def try_process(self, cmd):
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
     
    def setContext(self, data):
        if len(self.context) == 100:
            self.context.pop(0)
        self.context.append(data)
        self.session.append(data)

        self.save()
        
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
   
    def show_process(self, value=""):
        if self.show_process_state: print(value)

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
##            
##        for m in out.split():
##            for dm in dataMap:
##                if " "+m+" " in " "+dm[1]+" " and dm[1] != "":
##                    out = out.replace(" "+m+" ", " "+dm[0]+" ", 1)
##                    
        return output
    
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

    def write(self, filename, datalist=[]):
        try:
            file = open("memory/console/" + str(filename) + ".txt", "w")
            file.writelines(datalist)
            file.close()
        except Exception as e:
            print(e)

    def read_data(self, filename):
        return self.read("data/"+str(self.memory.index(filename)))

    def read_datafreq(self, filename):
        return self.read("datafreq/"+str(self.memory.index(filename)))

    def readfreq(self, data):
        return self.freq[self.memory.index(data)]
            
    def read(self, filename):
        file = open("memory/console/" + str(filename) + ".txt", "r")
        r = file.readlines()
        file.close()
        r = [x.replace("\n","") for x in r]
        return r

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

    def sort_last(self, li):
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
    
    def getRelated(self, data, sep="", treshold = 0.0, strict=False, length=False, db=None):
        infl = {}
        if db == None: db = self.memory
        for val in db:
            rel = self.getRelation(data, val, sep)
            if rel > treshold: infl.setdefault(val, rel)
            elif length != False:
                if sep == "":
                    if len(val) == length:
                        infl.setdefault(val, rel)
                else:
                    if len(val.split()) == length:
                        infl.setdefault(val, rel)
                        
        infl_keys = [x for x in infl.keys()]
        infl_values = [x for x in infl.values()]
        li = []
        for i in range(len(infl_values)):
            li.append((infl_keys[i], infl_values[i]))

        if strict:
            if sep == "": score = self.andClass(data)
            else: score = self.getScore([self.getAllClasses([data])])
            
            infl_i = self.getClassScore(infl, score, sep=sep)
            pd = self.sort_dict(infl_i)
            if len(pd) > 0 and pd[0][0] != data and len(pd) != 1:

                m = 0
                if pd[0][0] == data and len(pd) > 1: m = 1
                
                infl = {x:infl[x] for x in infl_i if infl_i[x] >= pd[m][-1]*0.70}# and infl_i[x] > 0}
        return infl
        
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

    def in_list(self, data, li, Type):
        if Type == 'any':
            return any([data in a for a in li])
        else:
            return all([data in a for a in li])
        
    def tryread(self, filename):
        try:
            return self.read(filename)
        except Exception as e:
            return []
        
    def saveData(self, filename, data):
        data = [x + "\n" for x in data]
        self.write(filename, data)

    def saveMethod(self, objectname, value=""):
        name = str(len(self.codebase))

        #save input to  memory
        self.write("codebase/"+name)
        self.save2codebase(objectname, value)
        self.setValue(name, objectname, value)
        
    def setValue(self, name, objectname, value):
        data = self.read("codebase/"+name)
        if len(data) == 0:
            data.append(objectname)
            data.append(value)
        self.saveData("codebase/"+name, data)

    def saveInput(self, objectname):
        if objectname not in self.memory:
            #save input to  memory
            for x in objectname.split():
                self.save2memory(x, 1)
            self.save2memory(objectname)
        self.write("data/"+str(self.memory.index(objectname)))
        self.write("datafreq/"+str(self.memory.index(objectname)))

    def setReply(self, objectname, reply):
        self.setContext(reply)
        if reply not in self.memory:
            #save input to  memory
            self.save2memory(reply)
        data = self.read("data/"+str(self.memory.index(objectname)))
        datafreq = self.read("datafreq/"+str(self.memory.index(objectname)))
        if reply in data:
            datafreq[data.index(reply)] = str(int(datafreq[data.index(reply)]) + 1)
        else:
            data.append(reply)
            datafreq.append("1")
        self.saveData("data/"+str(self.memory.index(objectname)), data)
        self.saveData("datafreq/"+str(self.memory.index(objectname)), datafreq)
        
    def save(self):
        self.saveData("memory", self.memory)
        self.saveData("codebase", [str(len(self.codebase))+"\n"])
        self.saveData("context", self.context)
        self.saveData("session", self.session)
        self.saveData("freq", self.freq)
        self.saveData("ifreq", self.ifreq)
        self.saveData('events', self.events_id)
        self.loadMemory()

    def load(self, filename):
        var = [x.replace("\n", "") for x in self.read(filename)]
        return var

    def loadContext(self):
        self.context = self.load("context")

    def loadCodebase(self):
        cb = int(open("memory/console/codebase.txt").read().replace("\n", ""))
        self.codebase = {open('memory/console/codebase/'+str(x)+'.txt').readlines()[0].replace("\n", "")
                         :open('memory/console/codebase/'+str(x)+'.txt').readlines()[1].replace("\n", "") for x in range(cb)}
        self.rev_codebase = {self.codebase[x]:x for x in self.codebase}
        
    def loadEvents(self):
        for r, d, f in os.walk('memory/console/events/'):
            pass
        self.events_id = {x.replace('.txt', '', 1).replace('\n', '', 1):open('memory/console/events/'+x).readlines()[0].replace('\n', '', 1) for x in f}
        self.events = {}
        for x in self.events_id:
            key = self.read_event(x, 'classes')
            if key in self.events:
                self.events[key].append(x)
            else:
                self.events.setdefault(key, [x])

    def info(self, string): #to show information
        if self.show_info:
            print(string)
        
    def loadMemory(self):
        if os.path.exists('memory/console/memory.txt'):
            self.loadContext()
            self.loadCodebase()
            
            try:
                self.loadEvents()
            except Exception as e:
                print(e)
            
            self.session = self.load("session")
            self.memory = self.load("memory")
            self.freq = self.load("freq")
            self.ifreq = self.load("ifreq")
        else:
            self.setup()
            self.createMemory()

    def setup(self, li=None):
        if li == None:
            li = ['resources', 'memory', 'sessions', 'memory/console/', 'memory/console/data/', 'memory/console/datafreq', 'memory/console/codebase', 'memory/console/events/']
        for x in li:
            if os.path.exists(x):
                pass
            else:
                self.info('setting up "{}"'.format(x))
                os.mkdir(x)

    def percentage_similarity(self, data, string, sep="", strict=True):
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
    
    def getRelation(self,data,string,sep=""):
        infl1 = self.percentage_similarity(data, string, sep)
        infl2 = self.percentage_similarity(string, data, sep)
        return formatVal((infl1*infl2))

    def sort_dict(self, dic, ascending=True):
        new = []
        if len(dic) > 0:
            d = dic.copy()
            while len(d) > 0:
                v = max(zip(d.values(), d.keys()))
                new.append((v[1],v[0]))
                d.pop(v[1])
        return new
        
    def getAllClasses(self, li):
        allclasses = []
        for x in li:
            for y in x.split(" "):
                l = self.getClasses(y)
                for a in l:
                    if a not in allclasses:
                        allclasses.append(a)
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
    
    def getQueAns(self, datalist):
        que = []
        ans = []
        infl = []
        for x in datalist:
            for y in self.read_data(x):
                if len(y) > 0:
                    ans.append(y)
                    que.append(x)
                    infl.append(int(self.read_datafreq(x)[self.read_data(x).index(y)]))
        return que, ans, infl
