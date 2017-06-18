
from functions import *

class brain_functions:
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
                
    def bestRelated(self, data, multiply_factor=0.0, strict=True, treshold=0.0, length=False, sep=""):
        rel = self.getRelated(data, sep, treshold, strict, len(data))
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
                if all([x in cl for cl in li]):
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
        classes = []
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
        if len(s1) == len(s2):
            l = [s1[x] for x in range(len(s1))]
            lmap = [s2[x] for x in range(len(s1))]
        else:
            for x in range(len(s)):
                if s[x] in s2:
                    
                    if abs(n-s1.index(s[x])) > 0 and len(" ".join(s1[n+1:s1.index(s[x])]).replace("`","").strip()) > 0:
                        
                        if s1.index(s[x]) == len(s1)-1:
                            l.append(" ".join(s1[n+1:]).strip())
                            lmap.append(" ".join(s2[n2+1:]).strip())
                            
                        else:
                            l.append(" ".join(s1[n+1:s1.index(s[x])]).strip())
                            lmap.append(" ".join(s2[n2+1:s2.index(s[x])]).strip())
                            
                        for vn in range(n2+1,s2.index(s[x]),1):
                            s2[vn] = "`"
                        for vn in range(n+1,s1.index(s[x]),1):
                            s1[vn] = "`"
                    else:
                        if s1.index(s[x]) == len(s1)-1:
                            l.append(" ".join(s1[n+1:]).strip())
                            lmap.append(" ".join(s2[n2+1:]).strip())
                            
                        else:
                            l.append(s[x])
                            lmap.append(s[x])
                            
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
        for d in dataMap:
            out = out.replace(" "+d[-1]+" ", " "+d[0]+" ")
##            
##        for m in out.split():
##            for dm in dataMap:
##                if " "+m+" " in " "+dm[1]+" " and dm[1] != "":
##                    out = out.replace(" "+m+" ", " "+dm[0]+" ", 1)
##                    
        return out.strip()
    
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
    
    def getRelated(self, data, sep="", treshold = 0.0, strict=False, length=False):
        infl = {}
        for val in self.memory:
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
            if pd[0][0] != data and len(pd) != 1:

                m = 0
                if pd[0][0] == data and len(pd) > 1: m = 1
                
                infl = {x:infl[x] for x in infl_i if infl_i[x] >= pd[m][-1]*0.70}# and infl_i[x] > 0}
        return infl
        
    def getScore(self, classes):
        d = {}
        l = [len(x) for x in classes]
        i = l.index(max(l))
        for x in classes[i]:
            if all([x in a for a in classes]):
                if x in d:
                    d[x] += 1
                else:
                    d.setdefault(x,1)
        tot = sum([d[x] for x in d])
        d = {x:d[x]/tot for x in d}
        return d
    
    def tryread(self, filename):
        try:
            return self.read(filename)
        except Exception as e:
            return []
        
    def saveData(self, filename, data):
        data = [x + "\n" for x in data]
        self.write(filename, data)

    def saveMethod(self, objectname, value=""):
        if objectname not in self.codebase:
            #save input to  memory
            self.save2codebase(objectname)
            self.write("codebase/"+objectname)
        self.setValue(objectname, value)

    def setValue(self, objectname, value):
        data = self.read("codebase/"+objectname)
        data.append(value)
        self.saveData("codebase/"+objectname, data)

    def saveInput(self, objectname):
        self.write("data/"+str(self.memory.index(objectname)))
        self.write("datafreq/"+str(self.memory.index(objectname)))

    def setReply(self, objectname, reply):
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
        self.saveData("codebase", self.codebase)
        self.saveData("context", self.context)
        self.saveData("session", self.session)
        self.saveData("freq", self.freq)        
        self.loadMemory()

    def load(self, filename):
        var = [x.replace("\n", "") for x in self.read(filename)]
        return var

    def loadContext(self):
        self.context = self.load("context")

    def loadMemory(self):
        self.loadContext()
        self.session = self.load("session")
        self.memory = self.load("memory")
        self.codebase = self.load("codebase")
        self.freq = self.load("freq")

    def getRelation(self,data,string,sep=""):
        if sep != "":
            d = data.split(sep)
            s = string.split(sep)
        else:
            d = list(data)
            s = list(string)
        c = 0
        for v in d:
            if v in s:
                if d.index(v) == s.index(v):
                    c += 1.0
                else:
                    c += 0.5
                s[s.index(v)] = "`"
                d[d.index(v)] = "`"
        if len(s) > 0:
            infl1 = c/len(s)
        else:
            infl1 = 0
        
        if sep != "":
            d = data.split(sep)
            s = string.split(sep)
        else:
            d = list(data)
            s = list(string)
        c = 0
        for v in s:
            if v in d:
                if d.index(v) == s.index(v):
                    c += 1.0
                else:
                    c += 0.5
                s[s.index(v)] = "`"
                d[d.index(v)] = "`"
        if len(d) > 0: 
            infl2 = c/len(d)
        else:
            infl2 = 0
        return formatVal((infl1*infl2))

    def sort_dict(self, dic, ascending=True):
        new = []
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
        for x in datalist:
            for y in self.read_data(x):
                if len(y) > 0:
                    ans.append(y)
                    que.append(x)
        return que, ans
