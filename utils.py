#!/usr/bin/python3
# coding=utf-8
import numpy as np
import os
from deal_image import *
import pathlib
# 创建一个schedule，里面包含epoch，一次epoch含有的数据量，batch大小

# 训练需要不停循环，使用for循环，那这个schedule要能返回迭代次数

# 1e5 1e3 

# 唯一的问题就是这个ordered_train_file可能太大了

def mylistCopy(in_lis):
    out_lis = []
    for i in in_lis:
        out_lis.append(i)
    return out_lis


class TrainSchedule:
    def __init__(self, epoch, file_paths, batch_size, func):
        self.epoch = epoch
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.data_num = len(self.file_paths) * self.epoch
        self._func = func
        self._build_schedule()
        self._get_iteration_num()
    
    def _build_schedule(self):
        
        self.ordered_train_file = []
        for i in range(self.epoch):
            # unit_epoch = self.file_paths.copy()
            unit_epoch = mylistCopy(self.file_paths)
            np.random.shuffle(unit_epoch)
            self.ordered_train_file += unit_epoch
    def _get_iteration_num(self):
        self.complete_batch_num = self.data_num // self.batch_size
        self.incomplete_data_num = self.data_num % self.batch_size
        self.iteration_num = self.complete_batch_num + (0 if self.incomplete_data_num == 0 else 1)
    def get_train_data(self, i, *args, **kargs):
        if i < self.complete_batch_num:
            return self._func(self.ordered_train_file[i*self.batch_size: (i+1)*self.batch_size], *args, **kargs)
        else:
            return self._func(self.ordered_train_file[i*self.batch_size: ], *args, **kargs)

def _loadSH(dirname,filename):
    fname = dirname+filename.split('_')[0]+'_'+filename.split('_')[1]+'_'+filename.split('_')[2]+".txt"
    #print("LOAD SH:  "+fname)
    this_sh = []
    fo = open(fname, "r")
    for line in fo.readlines():
        line = line.split()
        for i in range(len(line)):
            this_sh.append(line[i])
    return this_sh[0:3],this_sh[3:]

def read_img_from_files(files, dir=None, *args, **kargs):
    if dir:
        file_paths = [os.path.join(dir, f) for f in files]
    else:
        file_paths = files
    x = np.array([resize_image(imread(fp), *args, **kargs) for fp in file_paths])
    y = [_loadSH('data2/coefs/', pathlib.Path(fp).name) for fp in file_paths]
    y = np.array([t[0] for t in y])
    return x, y
    