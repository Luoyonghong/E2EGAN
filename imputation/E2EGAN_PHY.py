#-*- coding: utf-8 -*-
from __future__ import division
import os
import math
import time
import tensorflow as tf
import numpy as np
from ops import *
from utils import *
from tensorflow.python.ops import math_ops
from GRUI import mygru_cell

"""
生成器太弱，loss一直为负
尝试改成N次生成器，一次判别器
"""
class E2EGAN(object):
    model_name = "E2EGAN_Phy"     # name for checkpoint

    def __init__(self, sess, args, datasets):
        self.sess = sess
        self.isbatch_normal=args.isBatch_normal
        self.isNormal=args.isNormal
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name=args.dataset_name
        self.run_type=args.run_type
        self.lr = args.lr                 
        self.epoch = args.epoch     
        self.batch_size = args.batch_size
        self.n_inputs = args.n_inputs                 # MNIST data input (img shape: 28*28)
        self.n_steps = datasets.maxLength                                # time steps
        self.n_hidden_units = args.n_hidden_units        # neurons in hidden layer
        self.run_type=args.run_type
        self.result_path=args.result_path
        self.model_path=args.model_path
        self.pretrain_epoch=args.pretrain_epoch
        self.impute_iter=args.impute_iter
        self.isSlicing=args.isSlicing
        self.g_loss_lambda=args.g_loss_lambda
        
        self.datasets=datasets
        self.z_dim = args.z_dim         # dimension of noise-vector
        self.gen_length=args.gen_length
        
        # WGAN_GP parameter
        self.lambd = 0.25       # The higher value, the more stable, but the slower convergence
        self.disc_iters = args.disc_iters     # The number of critic iterations for one-step of generator

        # train
        self.learning_rate = args.lr
        self.beta1 = args.beta1
        if "1.5" in tf.__version__ or "1.7" in tf.__version__ :
            self.grui_cell_g1 = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grui_cell_g2 = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grui_cell_d = mygru_cell.MyGRUCell15(self.n_hidden_units)
        elif "1.4" in tf.__version__:
            self.grui_cell_g1 = mygru_cell.MyGRUCell4(self.n_hidden_units)
            self.grui_cell_g2 = mygru_cell.MyGRUCell4(self.n_hidden_units)
            self.grui_cell_d = mygru_cell.MyGRUCell14(self.n_hidden_units)
        elif "1.2" in tf.__version__:
            self.grui_cell_d = mygru_cell.MyGRUCell2(self.n_hidden_units)
            self.grui_cell_g1 = mygru_cell.MyGRUCell2(self.n_hidden_units)
            self.grui_cell_g2 = mygru_cell.MyGRUCell12(self.n_hidden_units)
        # test
        self.sample_num = 64  # number of generated images to be saved

        self.num_batches = len(datasets.x) // self.batch_size

      
    def pretrainG(self, x, m, ita, deltaPre, X_lengths, Keep_prob, is_training=True, reuse=False):

        with tf.variable_scope("g_enerator", reuse=reuse):
            #gennerate 
            z = self.time_series_to_z(x, m, ita, deltaPre, X_lengths, Keep_prob, reuse)
            #自编码器，x映射成z

            wr_h=tf.get_variable("g_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out= tf.get_variable("g_w_out",shape=[self.n_hidden_units, self.n_inputs],initializer=tf.random_normal_initializer())
            br_h= tf.get_variable("g_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out= tf.get_variable("g_b_out",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))
            w_z=tf.get_variable("g_w_z",shape=[self.z_dim,self.n_inputs],initializer=tf.random_normal_initializer())
            b_z=tf.get_variable("g_b_z",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))
            
            #self.times=tf.reshape(self.times,[self.batch_size,self.n_steps,self.n_inputs])
            #change z's dimension
            # batch_size*z_dim-->batch_size*n_inputs
            x=tf.matmul(z,w_z)+b_z
            x=tf.reshape(x,[-1,self.n_inputs])
            delta_zero=tf.constant(0.0,shape=[self.batch_size,self.n_inputs])
            #delta_normal=tf.constant(48.0*60.0/self.gen_length,shape=[self.batch_size,self.n_inputs])
            #delta:[batch_size,1,n_inputs]
            

            # combine X_in
            rth= tf.matmul(delta_zero, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            x=tf.concat([x,rth],1)
            
            X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])
            
            init_state = self.grui_cell_g2.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            #z=tf.reshape(z,[self.batch_size,1,self.z_dim])
            seq_len=tf.constant(1,shape=[self.batch_size])
            
            outputs, final_state = tf.nn.dynamic_rnn(self.grui_cell_g2, X_in, \
                                initial_state=init_state,\
                                sequence_length=seq_len,
                                time_major=False)
            init_state=final_state
            #outputs: batch_size*1*n_hidden
            outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
            # full connect
            out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
            out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])
            
            total_result=tf.multiply(out_predict,1.0)
            
            for i in range(1,self.n_steps):
                out_predict=tf.reshape(out_predict,[self.batch_size,self.n_inputs])
                #输出加上noise z
                #out_predict=out_predict+tf.matmul(z,w_z)+b_z
                #
                delta_normal=tf.reshape(self.imputed_deltapre[:,i:(i+1),:],[self.batch_size,self.n_inputs])
                rth= tf.matmul(delta_normal, wr_h)+br_h
                rth=math_ops.exp(-tf.maximum(0.0,rth))
                x=tf.concat([out_predict,rth],1)
                X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])
                
                outputs, final_state = tf.nn.dynamic_rnn(self.grui_cell_g2, X_in, \
                            initial_state=init_state,\
                            sequence_length=seq_len,
                            time_major=False)
                init_state=final_state
                outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
                out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
                out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])
                total_result=tf.concat([total_result,out_predict],1)
            
            #delta:[batch_size,,n_inputs]
        
            if self.isbatch_normal:
                with tf.variable_scope("g_bn", reuse=tf.AUTO_REUSE):
                    total_result=bn(total_result,is_training=is_training, scope="g_bn_imple")
            
            
            #last_values=tf.multiply(total_result,1)
            #sub_values=tf.multiply(total_result,1)

            return total_result 


    def discriminator(self, X, M, DeltaPre, Lastvalues ,DeltaSub ,SubValues , Mean,  X_lengths,Keep_prob, is_training=True, reuse=False, isTdata=True):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("d_iscriminator", reuse=reuse):
            
            wr_h = tf.get_variable("d_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out = tf.get_variable("d_w_out",shape=[self.n_hidden_units, 1],initializer=tf.random_normal_initializer())
            br_h = tf.get_variable("d_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out = tf.get_variable("d_b_out",shape=[1, ],initializer=tf.constant_initializer(0.001))
          
           
            M=tf.reshape(M,[-1,self.n_inputs])
            X = tf.reshape(X, [-1, self.n_inputs])
            DeltaPre=tf.reshape(DeltaPre,[-1,self.n_inputs])
           
            
            rth= tf.matmul(DeltaPre, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            # add noise
            #X=X+np.random.standard_normal(size=(self.batch_size*self.n_steps, self.n_inputs))/100 
            X=tf.concat([X,rth],1)
              
            X_in = tf.reshape(X, [self.batch_size, self.n_steps , self.n_inputs+self.n_hidden_units])
            
            init_state = self.grui_cell_d.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            outputs, final_state = tf.nn.dynamic_rnn(self.grui_cell_d, X_in, \
                                initial_state=init_state,\
                                sequence_length=X_lengths,
                                time_major=False)
         
            # final_state:batch_size*n_hiddensize
            # 不能用最后一个，应该用第length个  之前用了最后一个，所以输出无论如何都是b_out
            out_logit=tf.matmul(tf.nn.dropout(final_state,Keep_prob), w_out) + b_out
            out =tf.nn.sigmoid(out_logit)    #选取最后一个 output
            return out,out_logit


    def time_series_to_z(self, x, m, ita, deltaPre,X_lengths, keep_prob, reuse):
        with tf.variable_scope("g_TimeSeriesToZ", reuse=reuse):
            #缩写成ts2z
            #gennerate 
            wr_h = tf.get_variable("g_ts2z_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out = tf.get_variable("g_ts2z_w_out",shape=[self.n_hidden_units, self.z_dim],initializer=tf.random_normal_initializer())
            br_h = tf.get_variable("g_ts2z_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out = tf.get_variable("g_ts2z_b_out",shape=[self.z_dim, ],initializer=tf.constant_initializer(0.001))
          
            
            x = x + ita 
            #ita 应该用numpy初始化为很小的正态分布
            m = tf.reshape(m,[-1,self.n_inputs])
            x = tf.reshape(x, [-1, self.n_inputs])
            deltaPre = tf.reshape(deltaPre,[-1,self.n_inputs])
           
            
            rth = tf.matmul(deltaPre, wr_h)+br_h
            rth = math_ops.exp(-tf.maximum(0.0,rth))
            # add noise
            #X=X+np.random.standard_normal(size=(self.batch_size*self.n_steps, self.n_inputs))/100 
            x = tf.concat([x,rth],1)
              
            x_in = tf.reshape(x, [self.batch_size, self.n_steps , self.n_inputs+self.n_hidden_units])
            
            #这里用的GRUI是g1
            init_state = self.grui_cell_g1.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            outputs, final_state = tf.nn.dynamic_rnn(self.grui_cell_g1, x_in, \
                                initial_state=init_state,\
                                sequence_length=X_lengths,
                                time_major=False)
         
            # final_state:batch_size*n_hiddensize
            # 不能用最后一个，应该用第length个  之前用了最后一个，所以输出无论如何都是b_out
            z_out = tf.matmul(tf.nn.dropout(final_state, keep_prob), w_out) + b_out

            return z_out 



    def generator(self, x, m, ita, deltaPre, X_lengths, Keep_prob, is_training=True, reuse=False):
        # x,delta,n_steps
        # z :[self.batch_size, self.z_dim]
        # first feed noize in rnn, then feed the previous output into next input
        # or we can feed noize and previous output into next input in future version
        with tf.variable_scope("g_enerator", reuse=reuse):
            #gennerate 
            z = self.time_series_to_z(x, m, ita, deltaPre, X_lengths, Keep_prob, reuse)
            #自编码器，x映射成z

            wr_h=tf.get_variable("g_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out= tf.get_variable("g_w_out",shape=[self.n_hidden_units, self.n_inputs],initializer=tf.random_normal_initializer())
            br_h= tf.get_variable("g_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out= tf.get_variable("g_b_out",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))
            w_z=tf.get_variable("g_w_z",shape=[self.z_dim,self.n_inputs],initializer=tf.random_normal_initializer())
            b_z=tf.get_variable("g_b_z",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))
            
            # batch_size*z_dim-->batch_size*n_inputs
            x=tf.matmul(z,w_z)+b_z
            x=tf.reshape(x,[-1,self.n_inputs])
            delta_zero=tf.constant(0.0,shape=[self.batch_size,self.n_inputs])

            # combine X_in
            rth= tf.matmul(delta_zero, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            x=tf.concat([x,rth],1)
            
            X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])
            
            init_state = self.grui_cell_g2.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            #z=tf.reshape(z,[self.batch_size,1,self.z_dim])
            seq_len=tf.constant(1,shape=[self.batch_size])
            
            outputs, final_state = tf.nn.dynamic_rnn(self.grui_cell_g2, X_in, \
                                initial_state=init_state,\
                                sequence_length=seq_len,
                                time_major=False)
            init_state=final_state
            #outputs: batch_size*1*n_hidden
            outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
            # full connect
            out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
            out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])
            
            total_result=tf.multiply(out_predict,1.0)
            
            for i in range(1,self.n_steps):
                out_predict=tf.reshape(out_predict,[self.batch_size,self.n_inputs])
                #输出加上noise z
                #out_predict=out_predict+tf.matmul(z,w_z)+b_z
                #
                delta_normal=tf.reshape(self.imputed_deltapre[:,i:(i+1),:],[self.batch_size,self.n_inputs])
                rth= tf.matmul(delta_normal, wr_h)+br_h
                rth=math_ops.exp(-tf.maximum(0.0,rth))
                x=tf.concat([out_predict,rth],1)
                X_in = tf.reshape(x, [-1, 1, self.n_inputs+self.n_hidden_units])
                
                outputs, final_state = tf.nn.dynamic_rnn(self.grui_cell_g2, X_in, \
                            initial_state=init_state,\
                            sequence_length=seq_len,
                            time_major=False)
                init_state=final_state
                outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
                out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
                out_predict=tf.reshape(out_predict,[-1,1,self.n_inputs])
                total_result=tf.concat([total_result,out_predict],1)
            
            #delta:[batch_size,,n_inputs]
        
            if self.isbatch_normal:
                with tf.variable_scope("g_bn", reuse=tf.AUTO_REUSE):
                    total_result=bn(total_result,is_training=is_training, scope="g_bn_imple")
            
            
            last_values=tf.multiply(total_result,1)
            sub_values=tf.multiply(total_result,1)

            return total_result,self.imputed_deltapre,self.imputed_deltasub,self.imputed_m,self.x_lengths,last_values,sub_values
       

    def reduce_dimension(self, x, shape):
        if shape == 1:
            #x：shape:[n_inputs]
            x_new = x[:self.n_inputs]
        if shape == 2:
            #x:shape [num, n_inputs]
            x_new = []
            for item in x:
                x_new.append(item[:self.n_inputs])
        if shape == 3:
            #x:shape [num, num1, n_inputs]
            x_new = []
            for item in x:
                x_new_new = []
                for jth in item:
                    x_new_new.append(jth[:self.n_inputs])
                x_new.append(x_new_new)
        return x_new 
            
    def build_model(self):
        
        self.keep_prob = tf.placeholder(tf.float32) 
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.mean = tf.placeholder(tf.float32, [self.n_inputs,])
        self.deltaPre = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.lastvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.deltaSub = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.subvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.x_lengths = tf.placeholder(tf.int32,  shape=[self.batch_size,])
        self.imputed_deltapre=tf.placeholder(tf.float32,  shape=[self.batch_size,self.n_steps,self.n_inputs])
        self.imputed_deltasub=tf.placeholder(tf.float32,  shape=[self.batch_size,self.n_steps,self.n_inputs])
        self.imputed_m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        #self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
        self.ita = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs]) 
        
        

        """ Loss Function """

        Pre_x = self.pretrainG(self.x, self.m, self.ita, self.deltaPre, self.x_lengths, self.keep_prob, is_training=True, reuse=False)
        self.pretrain_loss = tf.reduce_sum(tf.square(tf.multiply(Pre_x, self.m)-self.x)) / tf.cast(tf.reduce_sum(self.m),tf.float32)
        
        #discriminator( X, M, DeltaPre, Lastvalues ,DeltaSub ,SubValues , Mean,  X_lengths,Keep_prob, is_training=True, reuse=False, isTdata=True):
        
        D_real, D_real_logits = self.discriminator(self.x, self.m, self.deltaPre,self.lastvalues,\
                                                   self.deltaSub,self.subvalues,  self.mean,\
                                                       self.x_lengths,self.keep_prob, \
                                                      is_training=True, reuse=False,isTdata=True)

        #G return total_result,self.imputed_deltapre,self.imputed_deltasub,self.imputed_m,self.x_lengths,last_values,sub_values
        g_x,g_deltapre,g_deltasub,g_m,G_x_lengths,g_last_values,g_sub_values = self.generator(self.x, self.m, self.ita, self.deltaPre, self.x_lengths, self.keep_prob, is_training=True, reuse=True)
        

        D_fake, D_fake_logits = self.discriminator(g_x, g_m, g_deltapre, g_last_values,\
                                                   g_deltasub, g_sub_values, self.mean,\
                                                      G_x_lengths, self.keep_prob,
                                                      is_training=True, reuse=True ,isTdata=False)
        
        l2_loss = tf.reduce_sum(tf.square(tf.multiply(g_x, self.m) - tf.multiply(self.x, self.m))) / tf.cast(tf.reduce_sum(self.m),tf.float32)
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)
        g_loss =  -d_loss_fake + self.g_loss_lambda * l2_loss 
        #d_loss = d_loss_real + d_loss_fake 
        d_loss = d_loss_real - g_loss 


        self.imputed_x = tf.multiply(self.x, self.m) + tf.multiply(g_x, (1-self.m))
        self.l2_loss = l2_loss
        self.d_loss_fake = d_loss_fake 
        self.g_x = g_x
        self.d_loss = d_loss
        self.g_loss = g_loss 

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        print("d vars:")
        for v in d_vars:
            print(v.name)
        print("g vars:")
        for v in g_vars:
            print(v.name)
        
        
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # this code have used batch normalization, so the upside line should be executed
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                        .minimize(self.d_loss, var_list=d_vars)
            #self.d_optim=self.optim(self.learning_rate, self.beta1,self.d_loss,d_vars)
            #self.g_optim = tf.train.AdamOptimizer(self.learning_rate*self.disc_iters, beta1=self.beta1) \
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                        .minimize(self.g_loss, var_list=g_vars)
            self.g_pre_optim = tf.train.AdamOptimizer(self.learning_rate*2,beta1=self.beta1) \
                        .minimize(self.pretrain_loss, var_list=g_vars)
        
        print(0.01)
        self.clip_all_vals = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in t_vars]
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in d_vars]
        self.clip_G = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in g_vars]
        

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        g_l2_loss = tf.summary.scalar("g_l2_loss", l2_loss)

        g_pretrain_loss_sum=tf.summary.scalar("g_pretrain_loss", self.pretrain_loss)
        # final summary operations
        self.g_sum = tf.summary.merge([g_loss_sum, g_l2_loss])
        self.g_pretrain_sum=tf.summary.merge([g_pretrain_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum,d_loss_fake_sum, d_loss_sum])
        
    def pretrain(self, start_epoch, counter, start_time):
        
        if start_epoch < self.pretrain_epoch:
            #todo
            for epoch in range(start_epoch, self.pretrain_epoch):
            # get batch data
                self.datasets.shuffle(self.batch_size,True)
                idx=0
                #x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
                for data_x,data_y,data_mean,data_m,data_deltaPre,data_x_lengths,data_lastvalues,_,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub in self.datasets.nextBatch():
                    
                    data_x = self.reduce_dimension(data_x,3)
                    data_mean = self.reduce_dimension(data_mean,1)
                    data_m = self.reduce_dimension(data_m,3)
                    data_deltaPre = self.reduce_dimension(data_deltaPre,3)
                    data_lastvalues = self.reduce_dimension(data_lastvalues,3)
                    imputed_deltapre = self.reduce_dimension(imputed_deltapre,3)
                    imputed_m = self.reduce_dimension(imputed_m,3)
                    deltaSub = self.reduce_dimension(deltaSub,3)
                    subvalues = self.reduce_dimension(subvalues,3)
                    imputed_deltasub = self.reduce_dimension(imputed_deltasub,3)

                    ita = np.random.normal(0, 0.01, size=(self.batch_size, self.n_steps, self.n_inputs))
                    _, summary_str,preloss = self.sess.run([self.g_pre_optim, self.g_pretrain_sum, self.pretrain_loss], 
                                                feed_dict={
                                                          self.x: data_x,
                                                          self.m: data_m,
                                                          self.ita: ita,
                                                          self.deltaPre: data_deltaPre,
                                                          self.mean: data_mean,
                                                          self.x_lengths: data_x_lengths,
                                                          self.lastvalues: data_lastvalues,
                                                          self.deltaSub:deltaSub,
                                                          self.subvalues:subvalues,
                                                          self.imputed_m:imputed_m,
                                                          self.imputed_deltapre:imputed_deltapre,
                                                          self.imputed_deltasub:imputed_deltasub,
                                                          self.keep_prob: 0.5})
                    self.writer.add_summary(summary_str, counter)
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, pre_loss: %.8f ,counter:%4d" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, preloss, counter))
                    idx += 1
                    counter += 1


    def train(self):

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name+'/'+self.model_dir)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            #start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            start_batch_id=0
            #counter = checkpoint_counter
            counter=start_epoch*self.num_batches
            print(" [*] Load SUCCESS")
            return 
        else:
            # initialize all variables
            tf.global_variables_initializer().run()
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        
        self.pretrain(start_epoch,counter,start_time)
        if start_epoch < self.pretrain_epoch:
            start_epoch=self.pretrain_epoch
        
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            self.datasets.shuffle(self.batch_size,True)
            idx=0
            for data_x,data_y,data_mean,data_m,data_deltaPre,data_x_lengths,data_lastvalues,_,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub in self.datasets.nextBatch():
                
                data_x = self.reduce_dimension(data_x,3)
                data_mean = self.reduce_dimension(data_mean,1)
                data_m = self.reduce_dimension(data_m,3)
                data_deltaPre = self.reduce_dimension(data_deltaPre,3)
                data_lastvalues = self.reduce_dimension(data_lastvalues,3)
                imputed_deltapre = self.reduce_dimension(imputed_deltapre,3)
                imputed_m = self.reduce_dimension(imputed_m,3)
                deltaSub = self.reduce_dimension(deltaSub,3)
                subvalues = self.reduce_dimension(subvalues,3)
                imputed_deltasub = self.reduce_dimension(imputed_deltasub,3)

                ita = np.random.normal(0, 0.01, size=(self.batch_size, self.n_steps, self.n_inputs))
                #_ = self.sess.run(self.clip_D)
                _ = self.sess.run(self.clip_all_vals)
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                               feed_dict={
                                                          self.x: data_x,
                                                          self.m: data_m,
                                                          self.ita: ita,
                                                          self.deltaPre: data_deltaPre,
                                                          self.mean: data_mean,
                                                          self.x_lengths: data_x_lengths,
                                                          self.lastvalues: data_lastvalues,
                                                          self.deltaSub:deltaSub,
                                                          self.subvalues:subvalues,
                                                          self.imputed_m:imputed_m,
                                                          self.imputed_deltapre:imputed_deltapre,
                                                          self.imputed_deltasub:imputed_deltasub,
                                                          self.keep_prob: 0.5})
                self.writer.add_summary(summary_str, counter)

                # update D network
                if counter%self.disc_iters==0:
                    #batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], 
                                                feed_dict={
                                                          self.x: data_x,
                                                          self.m: data_m,
                                                          self.ita: ita,
                                                          self.deltaPre: data_deltaPre,
                                                          self.mean: data_mean,
                                                          self.x_lengths: data_x_lengths,
                                                          self.lastvalues: data_lastvalues,
                                                          self.deltaSub:deltaSub,
                                                          self.subvalues:subvalues,
                                                          self.imputed_m:imputed_m,
                                                          self.imputed_deltapre:imputed_deltapre,
                                                          self.imputed_deltasub:imputed_deltasub,
                                                          self.keep_prob: 0.5})
                    self.writer.add_summary(summary_str, counter)
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f,counter:%4d" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss,counter))
                    #debug 

                counter += 1

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, counter:%4d" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, g_loss, counter))

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0 :
                    sample = self.sess.run([self.imputed_x],
                                            feed_dict={
                                                   self.x: data_x,
                                                   self.m: data_m,
                                                   self.ita: ita,
                                                   self.deltaPre: data_deltaPre,
                                                   self.mean: data_mean,
                                                   self.x_lengths: data_x_lengths,
                                                   self.lastvalues: data_lastvalues,
                                                   self.deltaSub:deltaSub,
                                                   self.subvalues:subvalues,
                                                   self.imputed_m:imputed_m,
                                                   self.imputed_deltapre:imputed_deltapre,
                                                   self.imputed_deltasub:imputed_deltasub,
                                                   self.keep_prob: 0.5})
                    
                    self.writeG_Samples("G_sample_x",counter, sample)
                    
                idx+=1
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

        
        self.save(self.checkpoint_dir, counter)

    def imputation(self, dataset, isTrain ):
        self.datasets = dataset 
        start_time = time.time()
        self.datasets.shuffle(self.batch_size,True)
        batch_id = 1
        for data_x,data_y,data_mean,data_m,data_deltaPre,data_x_lengths,data_lastvalues,_,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub in self.datasets.nextBatch():
            
            data_x = self.reduce_dimension(data_x,3)
            data_mean = self.reduce_dimension(data_mean,1)
            data_m = self.reduce_dimension(data_m,3)
            data_deltaPre = self.reduce_dimension(data_deltaPre,3)
            data_lastvalues = self.reduce_dimension(data_lastvalues,3)
            imputed_deltapre = self.reduce_dimension(imputed_deltapre,3)
            imputed_m = self.reduce_dimension(imputed_m,3)
            deltaSub = self.reduce_dimension(deltaSub,3)
            subvalues = self.reduce_dimension(subvalues,3)
            imputed_deltasub = self.reduce_dimension(imputed_deltasub,3)

            ita = np.random.normal(0, 0.01, size=(self.batch_size, self.n_steps, self.n_inputs))
            #_ = self.sess.run(self.clip_D)
            _ = self.sess.run(self.clip_all_vals)
            gx, l2loss, p_fake, gloss = self.sess.run([self.imputed_x, self.l2_loss, self.d_loss_fake, self.g_loss],
                                           feed_dict={
                                                      self.x: data_x,
                                                      self.m: data_m,
                                                      self.ita: ita,
                                                      self.deltaPre: data_deltaPre,
                                                      self.mean: data_mean,
                                                      self.x_lengths: data_x_lengths,
                                                      self.lastvalues: data_lastvalues,
                                                      self.deltaSub:deltaSub,
                                                      self.subvalues:subvalues,
                                                      self.imputed_m:imputed_m,
                                                      self.imputed_deltapre:imputed_deltapre,
                                                      self.imputed_deltasub:imputed_deltasub,
                                                      self.keep_prob: 1})
            print("Batchid: [%2d]  time: %4.4f, l2_loss: %.8f, p_fake: %.8f, gloss: %.8f"  % ( batch_id, time.time() - start_time, l2loss, p_fake, gloss))
            self.save_imputation(gx, batch_id, data_x_lengths, imputed_deltapre, data_y, isTrain )   
            batch_id = batch_id + 1

        if not isTrain:
            self.run_grui()
        
    def run_grui(self):
        random.seed()
        gpu=random.randint(0,1)
        command="CUDA_VISIBLE_DEVICES="+str(gpu)+" python ../GRUI/Run_GAN_imputed.py"+\
        ' --data-path=\"../Gan_Imputation/imputation_train_results/WGAN_One_Stage/'+self.model_dir+ '\"' + ' --n-inputs=' + str(self.n_inputs) 
        os.system(command)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.epoch,self.disc_iters,self.n_inputs, 
            self.batch_size, self.z_dim,
            self.lr,self.impute_iter,
            self.isNormal,self.isbatch_normal,
            self.isSlicing,self.g_loss_lambda,
            self.beta1
            )


    def save_imputation(self,impute_out,batchid,data_x_lengths,imputed_delta,data_y,isTrain):
        #填充后的数据全是n_steps长度！，但只有data_x_lengths才是可用的
        if isTrain:
            imputation_dir="imputation_train_results"
        else:
            imputation_dir="imputation_test_results"
        
        if not os.path.exists(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir)):
            os.makedirs(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir))
            
        #write imputed data
        resultFile=open(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir,\
                                     "batch"+str(batchid)+"x"),'w')
        for length in data_x_lengths:
            resultFile.writelines(str(length)+",")
        resultFile.writelines("\r\n")
        # impute_out:ndarray
        for oneSeries in impute_out:
            resultFile.writelines("begin\r\n")
            for oneClass in oneSeries:
                for i in oneClass.flat:
                    resultFile.writelines(str(i)+",")
                resultFile.writelines("\r\n")
            resultFile.writelines("end\r\n")
        resultFile.close()
        
        #write imputed_delta imputed_delta:list
        resultFile=open(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir,\
                                     "batch"+str(batchid)+"delta"),'w')
        for oneSeries in imputed_delta:
            resultFile.writelines("begin\r\n")
            for oneClass in oneSeries:
                for i in oneClass:
                    resultFile.writelines(str(i)+",")
                resultFile.writelines("\r\n")
            resultFile.writelines("end\r\n")
        resultFile.close()
        
        #write y
        resultFile=open(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir,\
                                     "batch"+str(batchid)+"y"),'w')
        for oneSeries in data_y:
            #resultFile.writelines("begin\r\n")
            for oneClass in oneSeries:
                resultFile.writelines(str(oneClass)+",")
            resultFile.writelines("\r\n")
            #resultFile.writelines("end\r\n")
        resultFile.close()
        
    def writeG_Samples(self,filename,step,o):
        if not os.path.exists(os.path.join("G_results",\
                                     self.model_name,\
                                     self.model_dir)):
            os.makedirs(os.path.join("G_results",\
                                     self.model_name,\
                                     self.model_dir))
        resultFile=open(os.path.join("G_results",\
                                     self.model_name,\
                                     self.model_dir,\
                                     filename+str(step)),'w')
        for oneSeries in o:
            resultFile.writelines("begin\r\n")
            for oneClass in oneSeries:
                for i in oneClass.flat:
                    resultFile.writelines(str(i)+",")
                resultFile.writelines("\r\n")
            resultFile.writelines("end\r\n")
        resultFile.close()
    
    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name, self.model_dir )

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
