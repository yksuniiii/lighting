import os
import math
import numpy as np

def _loadSH(dirname,filename):
    fname = dirname+filename.split('_')[0]+'_'+filename.split('_')[1]+"_output.txt"
    print("LOAD SH:  "+fname)
    this_sh = []
    fo = open(fname, "r")
    for line in fo.readlines():
        line = line.split()
        for i in range(len(line)):
            this_sh.append(line[i])
    return this_sh[0:3],this_sh[3:]


def analySH(fname):
    print "========"
    print fname
    fo = open(fname, "r")
    this_sh = []
    fname_len_sh = []
    for line in fo.readlines():
        line = line.split()
        for i in range(len(line)):
            this_sh.append(line[i])
    print "========"
    this_sh1 = []
    this_sh2 = []
    this_sh3 = []
    this_sh4 = []
    this_sh5 = []
    for sh in this_sh:
        if (abs(float(sh))<0.001):
            this_sh1.append(sh)
        elif (abs(float(sh))<0.01):
            this_sh2.append(sh)
        elif(abs(float(sh))<0.1):
            this_sh3.append(sh)
        elif(abs(float(sh))<1):
            this_sh4.append(sh)
        else:
            this_sh5.append(sh)
    print len(this_sh1),this_sh1
    print len(this_sh2),this_sh2
    print len(this_sh3),this_sh3
    print len(this_sh4),this_sh4
    print len(this_sh5),this_sh5
    fname_len_sh.append(len(this_sh1))
    fname_len_sh.append(len(this_sh2))
    fname_len_sh.append(len(this_sh3))
    fname_len_sh.append(len(this_sh4))
    fname_len_sh.append(len(this_sh5))
    return fname_len_sh





def main():
    ROOT = os.path.dirname(__file__)
    DEFAULT_CONTENT = os.path.join(ROOT, 'data/coefs/')
    contents_dir = DEFAULT_CONTENT
    filelist = [f for f in os.listdir(contents_dir)]
    num_files = len(filelist)
    fnamelen_all = [0,0,0,0,0]
    N = 0
    for i in range(num_files):
        filename = os.path.join(contents_dir, filelist[i])
        if(i<400):
            N = N+1
            fnamelen = analySH(filename)
            fnamelen_all[0]+=fnamelen[0]
            fnamelen_all[1]+=fnamelen[1]
            fnamelen_all[2]+=fnamelen[2]
            fnamelen_all[3]+=fnamelen[3]
            fnamelen_all[4]+=fnamelen[4]
    print fnamelen_all
    N = float(N)
    print fnamelen_all[0]/N,fnamelen_all[1]/N,fnamelen_all[2]/N,fnamelen_all[3]/N,fnamelen_all[4]/N



def _sh_preprocess(this_sh):
    for i in range(len(this_sh[0])):
        if this_sh[0][i] < 0:
            this_sh[0][i] = (-1)*math.pow(abs(this_sh[0][i]),0.5)
        elif this_sh[0][i] > 0:
            this_sh[0][i] = math.pow(abs(this_sh[0][i]),0.5)
        else:
            this_sh[0][i] = 0
    return this_sh

def _sh_unpreprocess(this_sh):
    for i in range(len(this_sh[0])):
        if this_sh[0][i] < 0:
            this_sh[0][i] = (-1)*math.pow(abs(this_sh[0][i]),2)
        elif this_sh[0][i] > 0:
            this_sh[0][i] = math.pow(abs(this_sh[0][i]),2)
        else:
            this_sh[0][i] = 0
        this_sh[0][i] = round(this_sh[0][i], 4)
    return this_sh

def main2():
    sh1 = [[1.23,-0.05,1.66,0.01,-0.0001]]
    print sh1
    sh1 = _sh_preprocess(sh1)
    print "after process"
    print sh1
    sh1 = _sh_unpreprocess(sh1)
    print "after unprocess"
    print sh1


def _loadSH(dirname,filename):
    fname = dirname+filename
    #print("LOAD SH:  "+fname)
    this_sh = []
    fo = open(fname, "r")
    for line in fo.readlines():
        line = line.split()
        for i in range(len(line)):
            this_sh.append(float(line[i]))
    return this_sh[0:3],this_sh[3:]


def dealSH_first3(filename):
    contents_dir = "./output/"
    filelist = [f for f in os.listdir(contents_dir)]
    num_files = len(filelist)
    a = np.zeros((60,3))
    for i in range(num_files):
        #print filelist[i]
        imgnum = filelist[i].split('_')[-1]
        epochnum = filelist[i].split('_')[-2]
        fname = filelist[i].split('_')[-3]
        if fname!=filename:
            continue
        if imgnum == '399.txt':
            index = int(epochnum)
            (first3,last24) = _loadSH(contents_dir,filelist[i])
            a[index]=first3
    print a
            

        # filename = os.path.join(contents_dir, filelist[i])
        # (sh1,sh2) = _loadSH(contents_dir, filelist[i])
        # sh1_list.append(sh1)
    



if __name__ == '__main__':
    fname = "aavsyvbazlnkeu"
    #fname = "aagwtamlnkpomz"
    dealSH_first3(fname)
