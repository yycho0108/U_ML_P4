#!/usr/bin/python
from matplotlib import pyplot as plt
import numpy as np
import csv
from collections import Counter

default_sectorSize = 100

last = False

def plotcsv(name):
    with open(name + '.csv','rb') as f:
        reader = csv.reader(f)
        data = np.asarray(list(reader),dtype=np.float32)
        print 'avg', np.average(data)

        sectorSize = default_sectorSize
        if(len(data) < sectorSize):
            sectorSize = len(data)

        if last:
            data = data[-sectorSize:]

        sector = len(data)/sectorSize

        data_av = np.arange(sectorSize, dtype=np.float32)
        for i in range(sectorSize):
            print "processing {}".format(i)
            data_av[i] = np.average(data[i*sector:(i+1)*sector])

        time = np.arange(sectorSize)
        print("MAX : {}".format(max(data)))

        plt.plot(data_av,'o')

        deg = 1
        fit = np.polyfit(time,data_av,deg=deg)
        fit_line = np.zeros(time.shape)
        for i in range(deg+1):
            fit_line += fit[i] * time**(deg - i)

        plt.plot(time,fit_line,color='red')

        plt.title(name + ': over {} iterations'.format(len(data)))
        plt.savefig(name + '.eps', bbox_inches='tight')
        plt.show()

        if name == 'test':
            cnt = Counter([float(e) for e in data])
            k,v = zip(*cnt.items())
            plt.bar(k,v,width=10)
            plt.xticks(k)
            plt.title('Highest Tiles')
            plt.show()

plotcsv('score')
plotcsv('penalties')
plotcsv('losses')
