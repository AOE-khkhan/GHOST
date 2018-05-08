import json, time

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
  
def get_sf(val):
    st = list(format((val),".20f"))
    sf = 0
    for s in st:
        if s == '.':
            continue

        if s != '0':
            break
        
        else:
            sf += 1
    return sf

def read_json(path):
    file = open(path+'.json', 'r')
    content = file.read()
    file.close()
    return json.loads(content)
    
def sent_tokenize(string):
    return string.split(".")

r = time.time()
track_time = False
def tt(string=''):
    global r, track_time
    if track_time:
        x = time.time()
        print('{} in {} seconds'.format(string, round(x-r, 4)))
        r = x

def word_tokenize(string):
    return list(string)#.split()


def write_file(path, contents):
    file = open(path, 'w')
    file.write(contents)
    file.close()

def write_json(path, dictionary):
    content = json.dumps(dictionary)
    file = open(path+'.json', 'w')
    file.write(content)
    file.close()
