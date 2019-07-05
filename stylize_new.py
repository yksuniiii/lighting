# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import os
from sys import stderr

import tensorflow as tf
import tflearn
import vgg
import mygen
from deal_image import *

_CONTENT_LAYER = 'relu4_2'
_weight_blend = [1,1,1,1,1]
content_shape = (1, 256, 256, 3)  # -> shape = [1, 512, 512, 3]

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
        print '%s[%d, %d]: [%.2f, %.2f, %.2f]' % (name, i, i,
                                                  image[i][i][0],
                                                  image[i][i][1],
                                                  image[i][i][2])


def print_progress(n_pic, n_iter, iterations, files):
    stderr.write('Iteration %d/%d ----- Image %d/%d\n' % (
        n_iter + 1, iterations, n_pic + 1, files))
'''
def stylize(dshape, iterations, contents_dir, tests_dir, logs_dir,ckpt_dir=None,output=None, device='/cpu'):
    network = tflearn.input_data(shape=[None, 256, 256, 3], name='input')
    content_net, mean_pixel = vgg.net(network)
    network = content_net[_CONTENT_LAYER]
    network = tflearn.fully_connected(network, 4096, activation='tanh')
    network = tflearn.fully_connected(network, 256, activation='tanh')
    network = tflearn.fully_connected(network, 27, activation='tanh')
    print network
    filelist = [f for f in os.listdir(contents_dir)
                        if os.path.splitext(f)[1][1:] == 'jpg' or  os.path.splitext(f)[1][1:] == 'JPEG' ]
    num_files = len(filelist)
    for i in range(num_files):
        print_progress(i, epoch - start, iterations, num_files)
        filename = os.path.join(contents_dir, filelist[i])
        content = imread(filename)
    X = # img set
    Y = # sh set

    model = tflearn.DNN(network, tensorboard_verbose=0) 
    model.fit({'input': X}, {'target': Y}, n_epoch=20,  
           validation_set=({'input': testX}, {'target': testY}),  
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')  
    return
'''

def _sh_loss(target_sh, my_sh):
    loss = tf.nn.l2_loss(target_sh-my_sh)
    return loss

def _loadSH(dirname,filename):
    fname = dirname+filename.split('_')[0]+'_'+filename.split('_')[1]+"_output.txt"
    print("LOAD SH:  "+fname)
    this_sh = []
    fo = open(fname, "r")
    for line in fo.readlines():
        line = line.split()
        for i in range(len(line)):
            this_sh.append(line[i])
    return this_sh


def _saveSH(testSH,outDir,filename,iters):
    print("SAVE SH: ",outDir,filename)
    fname = outDir+filename.split('_')[0]+'_'+filename.split('_')[1]+"_"+str(iters)+".txt"
    print("outfname = "+fname)
    print("testSH:")
    print testSH
    # fo = open(fname, "w")
    # fo.write()


def stylize(dshape, iterations, contents_dir, coefs_dir, tests_dir, logs_dir,ckpt_dir,
            output, device='/cpu'):
    print 'into the function stylize()'

    with tf.device(device):
        content_image = tf.placeholder('float', shape=content_shape)
        content_net, mean_pixel = vgg.net(content_image)
        content_feature = content_net[_CONTENT_LAYER]
        print content_feature
        mysh = tflearn.fully_connected(content_feature, 27, activation='linear')
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
                    if os.path.splitext(f)[1][1:] == 'jpg' or  os.path.splitext(f)[1][1:] == 'JPEG' ]
        num_files = len(filelist)
        print filelist
        filetest = [f for f in os.listdir(tests_dir)
                    if os.path.splitext(f)[1][1:] == 'jpg' or os.path.splitext(f)[1][1:] == 'JPEG']
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
                content_pre = np.array([vgg.preprocess(content, mean_pixel)])
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

