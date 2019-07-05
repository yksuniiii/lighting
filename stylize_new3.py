# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os
from sys import stderr

import tensorflow as tf
import tflearn
# import vgg
import mygen
from deal_image import *
import math
from collections import OrderedDict
import pathlib

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utils import TrainSchedule, read_img_from_files

_CONTENT_LAYER = 'relu4_2'
_weight_blend = [1,1,1,1,1]
content_shape = (None, 256, 256, 3)  # -> shape = [1, 512, 512, 3]

content_weight = 2
style_weight = 1
tv_weight = 1e-2

try:
    reduce
except NameError:
    from functools import reduce


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def _content_loss(content_net, net2):
    content_feature = content_net[_CONTENT_LAYER]
    _, height, width, number = map(lambda i: i.value,
                                   content_feature.get_shape())
    content_size = height * width * number
    loss = 1 * (
        2 * tf.nn.l2_loss(
            net2[_CONTENT_LAYER] - content_feature) / content_size)
    return loss


def print_pixels(image, name='image'):
    for i in range(40, 200, 40):
        print('%s[%d, %d]: [%.2f, %.2f, %.2f]') % (name, i, i,
                                                  image[i][i][0],
                                                  image[i][i][1],
                                                  image[i][i][2])


def print_progress(n_pic, n_iter, iterations, files):
    stderr.write('Iteration %d/%d ----- Image %d/%d\n' % (
        n_iter + 1, iterations, n_pic + 1, files))


def _sh_preprocess(this_sh):
    for i in range(len(this_sh[0])):
        if float(this_sh[0][i]) < 0:
            this_sh[0][i] = (-1)*math.pow(abs(float(this_sh[0][i])),0.5)
        elif float(this_sh[0][i]) > 0:
            this_sh[0][i] = math.pow(abs(float(this_sh[0][i])),0.5)
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



def _sh_loss(target_sh1, mysh1,target_sh2,mysh2):
    loss1 = tf.nn.l2_loss(target_sh1-mysh1)
    loss2 = tf.nn.l2_loss(target_sh2-mysh2)
    # total_loss = loss1*10+loss2
    total_loss = 100*loss1
    return (total_loss, loss1, loss2)

def _loadSH(dirname,filename):
    fname = dirname+filename.split('_')[0]+'_'+filename.split('_')[1]+filename.split('_')[2]+".txt"
    #print("LOAD SH:  "+fname)
    this_sh = []
    fo = open(fname, "r")
    for line in fo.readlines():
        line = line.split()
        for i in range(len(line)):
            this_sh.append(line[i])
    return this_sh[0:3],this_sh[3:]


def _saveSH(testSH1,testSH2,outDir,filename,iters,epc):
    print("SAVE SH: ",outDir,filename)
    fname = outDir+filename.split('_')[0]+'_'+filename.split('_')[1]+filename.split('_')[2]+"_"+str(epc)+'_'+str(iters)+".txt"
    print("outfname = "+fname)
    print("testSH1:")
    print(testSH1)
    print("testSH2:")
    print(testSH2)
    fo = open(fname, "w")
    for i in range(len(testSH1[0])):
        fo.write(str(testSH1[0][i]))
        if i%3==2:
            fo.write('\n')
        else:
            fo.write('\t')
    for i in range(len(testSH2[0])):
        fo.write(str(testSH2[0][i]))
        if i%3==2:
            fo.write('\n')
        else:
            fo.write('\t')
    fo.close()




def stylize(dshape, iterations, contents_dir, coefs_dir, tests_dir, logs_dir,ckpt_dir,
            output, device='/cpu'):
    print('into the function stylize()')

    with tf.device(device):
        content_image = tf.placeholder('float', shape=content_shape)
        mid1 = tflearn.conv_2d(content_image,64,3,strides = 2,activation='ReLU', 
        name='Layer1') # 64 (128, 128)
        mid2 = tflearn.conv_2d(mid1,128,3,strides = 2,activation='ReLU', 
        name='Layer2') # 128 (64, 64)
        mysh1_fork = tflearn.conv_2d(mid2, 256,3,strides = 2,activation='ReLU', 
        name='Layer3') # 256 (32, 32)
        
        # weights_init=tflearn.initializations.variance_scaling()
        ##########################################
        # make model more complicated
        '''
        mysh1 = tflearn.conv_2d(mysh1, 512,3,strides = 2,activation='relu',
                                weights_init=tflearn.initializations.xavier(False)) # 512 (16, 16)
        mysh1 = tflearn.conv_2d(mysh1, 512,3,strides = 1,activation='relu',
                                weights_init=tflearn.initializations.xavier(False)) # 512 (16, 16)
        mysh1 = tflearn.conv_2d(mysh1, 1024,3,strides = 2,activation='relu',
                                weights_init=tflearn.initializations.xavier(False)) # 1024 (8, 8)
        mysh1 = tflearn.conv_2d(mysh1, 1024,3,strides = 1,activation='relu',
                                weights_init=tflearn.initializations.xavier(False)) # 1024 (8, 8)
        '''
        ##########################################
        
        mysh2 = tflearn.conv_2d(mid2, 256,3,strides = 2,activation='ReLU')
        mysh1 = tflearn.fully_connected(mysh1_fork, 3, activation='linear',
                                        #weights_init=tflearn.initializations.variance_scaling(),
                                        name='Layer4')
        mysh2 = tflearn.fully_connected(mysh2, 24, activation='ReLU')
        ##########################################
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vs_reduced = []
        for v in vs:
            if 'Layer' in v.name:
                vs_reduced.append(v)
        print(len(vs_reduced))
        print(vs_reduced)
        ##########################################
        print("mysh1:")
        print(mysh1)
        print("mysh2:")
        print(mysh2)

        # load target sh
        target_sh1 = tf.placeholder('float')
        target_sh2 = tf.placeholder('float')
        print("target_sh1:")
        print(target_sh1)
        print("target_sh2:")
        print(target_sh2)
        (total_loss,loss1,loss2) = _sh_loss(target_sh1,mysh1,target_sh2,mysh2)
        print("total_loss:")
        print(total_loss)
        print("loss1:")
        print(loss1)
        print("loss2:")
        print(loss2)

        # optimizer set
        train_step = tf.train.AdamOptimizer().minimize(total_loss)
        # train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(total_loss)

    #mysh = tf.reduce_mean(mysh, name = 'my-sh')
    #tf.summary.scalar('my-sh', mysh)
 
    tf.summary.scalar('total-loss', total_loss)
    # tf.summary.scalar('loss1', loss1)
    # tf.summary.scalar('loss2', loss2)
    
    gd = tf.gradients(total_loss, vs_reduced)
    print(gd)
    tf.summary.histogram('layer1', mid1)
    tf.summary.histogram('layer2', mid2)
    tf.summary.histogram('layer3', mysh1_fork)
    tf.summary.histogram('output', mysh1)
    for i, e in enumerate(gd):
        if i % 2 == 0:
            tf.summary.histogram('gradient-w{}'.format(i), e)
        else:
            tf.summary.histogram('gradient-b{}'.format(i), e)
    for i, e in enumerate(vs_reduced):
        if i % 2 == 0:
            tf.summary.histogram('b{:d}'.format(i//2+1), e)
        else:
            tf.summary.histogram('w{:d}'.format((i+1)//2), e)
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        writer = tf.summary.FileWriter(logs_dir, sess.graph)
        merged = tf.summary.merge_all()

        
        sess.run(tf.initialize_all_variables())
        
        '''
        for i, e in enumerate(vs_reduced):
            w = sess.run(e)
            plt.hist(w.reshape(-1))
            plt.show()
        exit(0)
        '''
        
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("=== ckpt model ====")
            print(ckpt.model_checkpoint_path)
            print("=== ckpt model ====")
            saver.restore(sess, ckpt.model_checkpoint_path)

        start = global_step.eval()
        # start = 0
        print("start epoch:", start)
        print(contents_dir, tests_dir)
        filelist = [f for f in os.listdir(contents_dir)
                    if os.path.splitext(f)[1][1:] == 'png' or  os.path.splitext(f)[1][1:] == 'PNG' ]
        
        ##########################################
        '''
        file_names_front_part = ['_'.join(fn.split('_')[:2]) for fn in filelist]
        file_names_rest_part = ['_'.join(fn.split('_')[2:]) for fn in filelist]
        file_names_simple = list(OrderedDict.fromkeys(file_names_front_part))
        
        file_names_reduced = []
        j = 0
        for i, fn_sim in enumerate(file_names_simple):
            while fn_sim not in filelist[j]:
                j += 1
            file_names_reduced.append(filelist[j])
        print(len(file_names_reduced))
        filelist = file_names_reduced
        '''
        ##########################################
        
        num_files = len(filelist)
        filetest = [f for f in os.listdir(tests_dir)
                    if os.path.splitext(f)[1][1:] == 'png' or os.path.splitext(f)[1][1:] == 'PNG']
        num_tests = len(filetest)
   
        
        ##########################################
        '''
        for fn in filetest:
            for i, fn_train in enumerate(filelist):
                fn_name_without_num = '_'.join(fn.split('_')[:2])
                if '_'.join(fn.split('_')[:2]) in fn_train:
                    filelist[i] = fn
                    print(fn)
        '''
        if os.path.exists('mean_img.npy'):
            mean_img = np.load('mean_img.npy')
        else:
            img_means = [resize_image(imread(os.path.join(contents_dir, fn)), dshape) for fn in filelist]
            mean_img = np.mean(img_means, axis=0) / 255
            np.save('mean_img.npy', mean_img)
        ##########################################
        #mean_img = 0
        #+++++++++++++++++++++++++++++++++++++++++
        
        schedule = TrainSchedule(iterations, filelist, 1, read_img_from_files)
        
        #-----------------------------------------
        
        # iterations
        print(schedule.iteration_num, schedule.complete_batch_num, schedule.incomplete_data_num)
        for i in range(schedule.iteration_num):
            # train
            content_pre, cur_sh1 = schedule.get_train_data(i, dir=contents_dir, dshape=dshape)
            content_pre = content_pre.astype('float32') / 255 - mean_img
            if i == 0:
                print(np.mean(content_pre.reshape((-1, 1))), np.amax(content_pre.reshape((-1, 1))))
            '''
            plt.hist(content_pre.reshape((-1, 1)))
            plt.show()
            '''
            #get sh
            #cur_sh1 = _sh_preprocess(cur_sh1)
            cur_sh2 = 0
            # print "cur_sh1"
            # print cur_sh1
            # print "cur_sh2"
            # print cur_sh2
            feed_dict = {target_sh1: cur_sh1, target_sh2: cur_sh2, content_image: content_pre}

            _, w4, loss, loss_loss1, loss_loss2, mysh1_out, mysh2_out, summary = sess.run(
                [train_step, vs_reduced[6], total_loss, loss1, loss2, mysh1,mysh2, merged],
                feed_dict=feed_dict)
            
            if i < 20:
                plt.figure()
                plt.subplot(111)
                plt.boxplot(w4)
                plt.savefig(os.path.join('./FIG/act/', '{}.png'.format(i)))
                plt.close()
            # print "mysh1:"
            # print mysh1
            # print "mysh2:"
            # print mysh2
            # print  "mysh1_out:"
            # print mysh1_out
            # print  "mysh2_out:"
            # print mysh2_out
            # print "==== unprocess ===="
            # mysh1_out = _sh_unpreprocess(mysh1_out)
            mysh2_out = _sh_unpreprocess(mysh2_out)
            # print("mysh1_out:")
            # print(mysh1_out)
            # print  "mysh2_out:"
            # print mysh2_out


            print("iteration:%d/%d this_loss:%f  loss1:%f loss2:%f" % (i, schedule.iteration_num, loss, loss_loss1, loss_loss2))
            if loss > 100:
                print("current iter maybe unlucky, savepng")
                print(content_pre)
                # fo = open("errorimg_name.txt","a")
                # fo.write("loss:%f"%(loss,))

            if i < 10 or i % 30 == 0:
                writer.add_summary(summary, i)
            if (i+1) % 5000 == 0:
                print("save checkpoint")
                global_step.assign(i).eval()
                saver.save(sess, ckpt_dir + '/model.ckpt',
                           global_step=global_step)
           
            #save sh
            if (i+1)%200 == 0:
                for j in range(10):
                    filename = os.path.join(tests_dir, filetest[j])
                    fname, _ = os.path.splitext(os.path.basename(filename))
                    content = imread(filename)
                    print("filename:"+filename)
                    content = resize_image(content, dshape)
                    content_pre = (np.array([content / 255]) - mean_img)
                    feed_dict = {content_image: content_pre}
                    mysh1_out,mysh2_out = sess.run([mysh1,mysh2],feed_dict=feed_dict)
                    _saveSH(mysh1_out,mysh2_out,output,filetest[j],i,0)

                




''''
def stylize(dshape, iterations, contents_dir, coefs_dir, tests_dir, logs_dir,ckpt_dir,
            output, device='/cpu'):
    print 'into the function stylize()'

    with tf.device(device):
        content_image = tf.placeholder('float', shape=content_shape)
        
        print content_image
        mid1 = tflearn.conv_2d(content_image,64,3,strides = 2,activation='relu')
        print mid1
        mid2 = tflearn.conv_2d(mid1,128,3,strides = 2,activation='relu')
        print mid2
        mysh = tflearn.fully_connected(mid2, 27, activation='linear')
        mysh = 3*mysh
        print mysh
        

        # load target sh
        target_sh = tf.placeholder('float', shape=mysh.shape)
        print mysh.shape
        print target_sh
        total_loss = _sh_loss(target_sh,mysh)
        print total_loss

        # optimizer set
        train_step = tf.train.AdamOptimizer().minimize(total_loss)

    #mysh = tf.reduce_mean(mysh, name = 'my-sh')
    #tf.summary.scalar('my-sh', mysh)
    tf.summary.scalar('total-loss', total_loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logs_dir, sess.graph)
        merged = tf.summary.merge_all()

        sess.run(tf.initialize_all_variables())

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)

        start = global_step.eval()
        # start = 0
        print("start epoch:", start)
        print contents_dir, tests_dir
        filelist = [f for f in os.listdir(contents_dir)
                    if os.path.splitext(f)[1][1:] == 'png' or  os.path.splitext(f)[1][1:] == 'PNG' ]
        num_files = len(filelist)
        print filelist
        filetest = [f for f in os.listdir(tests_dir)
                    if os.path.splitext(f)[1][1:] == 'png' or os.path.splitext(f)[1][1:] == 'PNG']
        num_tests = len(filetest)
        print filetest
       
        # iterations
        for epoch in range(start, start + iterations):
            random.shuffle( filelist)
            # train
            for i in range(num_files):
                print_progress(i, epoch - start, iterations, num_files)
                filename = os.path.join(contents_dir, filelist[i])
                content = imread(filename)
                print "filename:"+filename
                content = resize_image(content, dshape)
                content_pre = np.array([content])
                #get sh
                cur_sh = _loadSH(coefs_dir,filelist[i])
                cur_sh = np.reshape(cur_sh, (1, 27))
                feed_dict = {target_sh: cur_sh, content_image: content_pre}

                _, loss, mysh_out, summary = sess.run(
                    [train_step, total_loss, mysh, merged],
                    feed_dict=feed_dict)
                print  ("mysh:")
                print mysh
                print  ("mysh_out:")
                print mysh_out
                if (i+1)%5 == 1 :
                    _saveSH(mysh_out,output,filelist[i],i)


                print "epoch=%d files=%d this_loss:%f " % (epoch, i, loss)
                writer.add_summary(summary, epoch * num_files + i)
                #save sh
                if (i+1)%20 == 0 or i == num_files-1:
                    for j in range(num_tests):
                        filename = os.path.join(tests_dir, filetest[j])
                        fname, _ = os.path.splitext(os.path.basename(filename))
                        test_sh_out = sess.run(mysh, feed_dict={target_sh: cur_sh})
                        _saveSH(test_sh_out,output,filename,i)

'''