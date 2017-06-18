def word_tokenize(string):
    return list(string)#.split()

def sent_tokenize(string):
    return string.split(".")

def formatVal(num):
    num = float(format((num),".3f"))
    s = str(num)
    if len(s[s.index(".")+1:]) < 3:
        return num
    elif len(s[s.index(".")+1:]) > 2 and s[-1] in [str(x) for x in range(5)]:
        return float(str(num)[:-1])
    else:
        num += 0.01
        return float(str(num)[:s.index(".")+3])
    
