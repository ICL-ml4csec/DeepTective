import pickle
import os

# This file creates the TestSet for testing performance of other tools

source = '../data/'
target = './vuln-funs-'
groups = ['sard','git','nvd']

for g in groups:
    f = open(source + g + '_data_raw.pkl', 'rb')
    data = pickle.load(f)
    path = target+g
    os.mkdir(path)
    n = 1
    for d in data:
        if d[1]:
            name = path + '/' + g + '_' + str(d[1]) + '_' + str(n) + '.php'
            with open(name, 'w') as f:
                f.write(d[0])
            n += 1
