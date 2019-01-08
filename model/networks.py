import tensorflow as tf
import json
import numpy as np
import data.loader as loader


class BLSTM:
    def __init__(self, xhc_size, lr=0.01):
        with tf.variable_scope('BLSTM', reuse=tf.AUTO_REUSE):

            self.flstm = LSTM(xhc_size, 'FLSTM')
            self.blstm = LSTM(xhc_size, 'BLSTM')

            # Error
            # self.input = tf.placeholder(dtype=tf.float32, shape=[None, xhc_size[0]], name='sentence')
            # self.label = tf.placeholder(dtype=tf.float32, shape=xhc_size[1], name='probability')
            #
            # self.fhout = self.flstm.forward(self.input)
            # self.bhout = self.blstm.forward(tf.reverse(self.input, 0))
            # self.hout = tf.reduce_mean(self.fhout, self.bhout)
            #
            # self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.hout))
            # self.optm = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)

            self.base_address = 'C:\\Users\\Dell\\Desktop\\UCAS\\大三上\\人工智能导论\\大作业\\Final-Project-of-AI\\'
            self.saver = tf.train.Saver(max_to_keep=3)

    def train(self, data_path, maxepoch, lr=0.01, continue_train=False, trained_steps=0):
        data = json.load(open(data_path, 'r'))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if continue_train:
                latest = tf.train.latest_checkpoint(self.base_address + 'parameters/')
                self.saver.restore(sess, latest)

            #  Epoch
            for j in range(maxepoch):
                #  Sample in a Epoch
                for i in range(np.shape(data)[0]):
                    indexes = data[i]['indexes']
                    times = data[i]['times']
                    attributes = data[i]['attributes']
                    values = data[i]['values']
                    results = data[i]['results']
                    ddata, label = loader.data_process(indexes, times, attributes, values, results)

                    loss_array = []
                    #  Each Sample has many combinations to input
                    for k in range(np.shape(ddata)[0]):
                        fhout = self.flstm.forward(ddata[k])

                        bdata = list(ddata[k])[-1:0:-1]
                        bdata.append(ddata[k][0])
                        bdata = np.array(bdata)
                        bhout = self.blstm.forward(bdata)

                        hout = tf.reduce_mean([fhout, bhout], 0)
                        loss = tf.nn.l2_loss(tf.subtract(hout, np.float32(label[k])))
                        optm = tf.train.GradientDescentOptimizer(lr).minimize(loss)

                        # Training while building graphics
                        loss, _ = sess.run([loss, optm], feed_dict={self.input: ddata[k], self.label: label})

                        # Has built a graphic, but not available now.
                        # loss, _ = sess.run([self.loss, self.optm], feed_dict={self.input: ddata[k], self.label: label})
                        loss_array.append(loss)

                    print('Epoch:%d  Sample:%d  Mean Loss:%05f' %(j, i, np.average(loss_array)),
                          'Loss: ', loss_array)
                self.saver.save(sess, 'parameters/BLSTM', global_step=trained_steps+j)


class LSTM:
    def __init__(self, xhc_size, name='LSTM'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            self.xhc_size = xhc_size

            self.input_gate = Gate(xhc_size, 'InputGate')
            self.forget_gate = Gate(xhc_size, 'ForgetGate')
            self.state_gate = Gate(xhc_size, 'State_Gate', tf.nn.tanh)
            self.output_gate = Gate(xhc_size, 'OutputGate', tf.nn.sigmoid, True)

    def forward(self, x):
        hout_array = []
        h = tf.zeros(shape=[1, self.xhc_size[1]], dtype=tf.float32)
        C = tf.zeros(shape=self.xhc_size[2], dtype=tf.float32)
        insize = np.shape(x)[0]
        for i in range(insize):
            # Get the output of gates
            input = self.input_gate.forward(x[i], h)
            forget = self.forget_gate.forward(x[i], h)
            state = self.state_gate.forward(x[i], h)

            # Renew the state of LSTM
            C = tf.add(tf.multiply(input, state), tf.multiply(forget, C))
            o = self.output_gate.forward(x[i], h, C)
            h = tf.multiply(o, tf.nn.tanh(tf.reduce_mean(C)))
            hout_array.append(h)
        hout = tf.reduce_mean(hout_array)
        return hout


class Gate:
    def __init__(self, xhc_size, name='gate', activation=tf.nn.sigmoid, outgate=False):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            xsize = xhc_size[0]
            hsize = xhc_size[1]
            Csize = xhc_size[2]
            Osize = xhc_size[2] if not outgate else hsize
            self.W = tf.get_variable(name='W', shape=[xsize, Osize], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())
            self.U = tf.get_variable(name='U', shape=[hsize, Osize], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())
            self.b = tf.get_variable(name='b', shape=[Osize], dtype=tf.float32,
                                     initializer=tf.random_normal_initializer())
            self.activation = activation
            self.outgate = outgate
            if outgate:
                self.V = tf.get_variable(name='V', shape=[Csize, Osize], dtype=tf.float32,
                                         initializer=tf.random_normal_initializer())

    def forward(self, x, h, C=None):
        output = tf.add(tf.matmul(x[np.newaxis, :], self.W), tf.matmul(h, self.U))
        output = tf.add(output, self.b)
        if C is None:
            assert (self.outgate==False), 'No state C input to output gate'
        else:
            assert (self.outgate==True), 'Input state C input a normal gate'
            output_c = tf.matmul(C, self.V)
            output = tf.add(output, output_c)
        return output
