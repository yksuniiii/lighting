import tensorflow as tf
import tflearn
from tflearn.layers.conv import avg_pool_2d

'''
    To see the struct go to ./pgg.jpg
        and network.jpg for prisma's
    for more details run this file and use:
        tensorboard --logdir='logs/nn_logs'
'''


# mean_pixel = [123.68, 116.779, 103.939]

def net(input_image):
    ratios = [32, 16, 8, 4, 2, 1]
    conv_num = 8
    network = []
    for i in range(len(ratios)):
        network.append(avg_pool_2d(input_image, ratios[i]))

        # block_i_0, block_i_1, block_i_2
        for block in range(3):
            with tf.name_scope('block_%d_%d' % (i, block)):
                filter_size = 1 if (block + 1) % 3 == 0 else 3
                network[i] = tflearn.conv_2d(network[i], nb_filter=conv_num,
                                             filter_size=filter_size,
                                             weights_init='xavier',
                                             name='Conv_%d_%d' % (i, block))
                network[i] = tf.nn.batch_normalization(network[i], 0, 1.0,
                                                       0.0, 1.0, 1e-5,
                                                       'BatchNorm')
                network[i] = tflearn.activations.leaky_relu(network[i])

        if i == 0:
            network[i] = tflearn.upsample_2d(network[i], 2)
        else:
            upnet = tf.nn.batch_normalization(network[i - 1], 0, 1.0, 0.0, 1.0,
                                              1e-5, 'BatchNorm')
            downnet = tf.nn.batch_normalization(network[i], 0, 1.0, 0.0, 1.0,
                                                1e-5, 'BatchNorm')
            # join_i
            network[i] = tflearn.merge([upnet, downnet], 'concat', axis=3)
            # block_i_3, block_i_4, block_i_5
            for block in range(3, 6):
                with tf.name_scope('block_%d_%d' % (i, block)):
                    filter_size = 1 if (block + 1) % 3 == 0 else 3
                    network[i] = tflearn.conv_2d(network[i],
                                                 nb_filter=conv_num * i,
                                                 filter_size=filter_size,
                                                 weights_init='xavier',
                                                 name='Conv_%d_%d' % (i, block))
                    network[i] = tf.nn.batch_normalization(network[i], 0, 1.0,
                                                           0.0, 1.0, 1e-5,
                                                           'BatchNorm')
                    network[i] = tflearn.activations.leaky_relu(network[i])

            if i != len(ratios) - 1:
                network[i] = tflearn.upsample_2d(network[i], 2)

    network[len(ratios) - 1] = tflearn.conv_2d(network[len(ratios) - 1],
                                               nb_filter=3,
                                               filter_size=1,
                                               weights_init='xavier',
                                               name='Conv2d_out')
    return network[len(ratios) - 1]
