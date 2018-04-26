class Subsetter:
    def getSubsets(self, data):
        classified  = []
        dataList = data.split()
        for i in dataList:
            if i not in classified: classified.append(i)
        li = [x for x in dataList]
        for i in range(len(dataList)-1):
            li.pop(-1)
            x = " ".join(li)
            if x not in classified: classified.append(x)
        li = [x for x in dataList]
        li.reverse()
        for i in range(len(dataList)-1):
            li.pop(-1)
            li.reverse()
            x = " ".join(li)
            if x not in classified: classified.append(x)
            li.reverse()
        c = 0
        li = [x for x in dataList]
        for i in range(len(dataList)):
            for n in range(i,len(dataList)):
                x = " ".join(li[i:n])
                if x not in classified: classified.append(x)
        while "" in classified: classified.remove("")

        if data not in classified: classified.append(data)
        
        return classified
