#-*- coding: utf-8 -*-
from __future__ import division
import os
import math
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from ops import *
from utils import *
from GRUI import mygru_cell
import numpy as np

"""
E^2EGAN for imputation 
"""
class E2EGAN_AQ(object):
    model_name = "E^2EGAN"     # name for checkpoint

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
        self.n_steps = args.n_steps                                # time steps
        self.missing_rate=args.missing_rate
        self.n_hidden_units = args.n_hidden_units        # neurons in hidden layer
        self.n_classes = args.n_classes                # MNIST classes (0-9 digits)
        self.run_type=args.run_type
        self.result_path=args.result_path
        self.model_path=args.model_path
        self.pretrain_epoch=args.pretrain_epoch
        self.isSlicing=args.isSlicing
        self.g_loss_lambda=args.g_loss_lambda
        self.model_name+="_"+str(args.missing_rate)
        self.datasets=datasets
        self.z_dim = args.z_dim         # dimension of noise-vector
        self.use_grui = args.use_grui 
        print(self.n_inputs) 
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

        self.num_batches = len(datasets.aq_train_data) // (self.batch_size*48)

      
    def pretrainG(self, x, m, ita, delta,  X_lengths, Keep_prob, is_training=True, reuse=False):
    
        with tf.variable_scope("g_enerator", reuse=reuse):
            #gennerate 
            z = self.time_series_to_z(x, m, ita, delta, X_lengths, Keep_prob, reuse)
            #自编码器，x映射成z
            
            wr_h=tf.get_variable("g_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out= tf.get_variable("g_w_out",shape=[self.n_hidden_units, self.n_inputs],initializer=tf.random_normal_initializer())
            
            br_h= tf.get_variable("g_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out= tf.get_variable("g_b_out",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))
            w_z=tf.get_variable("g_w_z",shape=[self.z_dim,self.n_inputs],initializer=tf.random_normal_initializer())
            b_z=tf.get_variable("g_b_z",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))
            
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
                delta_normal=tf.reshape(self.imputed_delta[:,i:(i+1),:],[self.batch_size,self.n_inputs])
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

            return total_result 
            


    def discriminator(self, X,  DeltaPre, X_lengths,Keep_prob, is_training=True, reuse=False, isTdata=True):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        with tf.variable_scope("d_iscriminator", reuse=reuse):
            
            wr_h=tf.get_variable("d_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out= tf.get_variable("d_w_out",shape=[self.n_hidden_units, 1],initializer=tf.random_normal_initializer())
            br_h= tf.get_variable("d_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out= tf.get_variable("d_b_out",shape=[1, ],initializer=tf.constant_initializer(0.001))
          
           
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
          
            
            #x = x + ita 
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

            return z_out + ita 


    def generator(self, x, m, ita, deltaPre, X_lengths,  Keep_prob, is_training=True, reuse=False):
        # x,delta,n_steps
        # z :[self.batch_size, self.z_dim]
        # first feed noize in rnn, then feed the previous output into next input
        # or we can feed noize and previous output into next input in future version
        with tf.variable_scope("g_enerator", reuse=reuse):
            #gennerate 
            
            z = self.time_series_to_z(x, m, ita, deltaPre, X_lengths, Keep_prob, reuse)

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
            delta_zero=tf.reshape(self.imputed_delta[:,0:1,:],[self.batch_size,self.n_inputs])
            
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
                delta_normal=tf.reshape(self.imputed_delta[:,i:(i+1),:],[self.batch_size,self.n_inputs])
                rth= tf.matmul(delta_normal, wr_h)+br_h
                rth=math_ops.exp(-tf.maximum(0.0,rth))
                #rth= tf.constant(1.0,shape=[self.batch_size,self.n_hidden_units])
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
            
            return total_result, self.imputed_delta, self.x_lengths
        
    def reduce_dimension(self, x, shape):
        n = self.n_inputs//2
        if shape == 1:
            #x：shape:[n_inputs]
            x_new = x[:n]
        if shape == 2:
            #x:shape [num, n_inputs]
            x_new = []
            for item in x:
                x_new.append(item[:n])
        if shape == 3:
            #x:shape [num, num1, n_inputs]
            x_new = []
            for item in x:
                x_new_new = []
                for jth in item:
                    x_new_new.append(jth[:n])
                x_new.append(x_new_new)
        return x_new 
            
    def build_model(self):
        
        self.keep_prob = tf.placeholder(tf.float32) 
        self.aq = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
        self.complete_aq = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
        self.aq_m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
        self.aq_delta = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
        self.aq_lastvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
    
        self.meo = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
        self.complete_meo = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
        self.meo_m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
        self.meo_delta = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
        self.meo_lastvalues = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs/2])
        
        #self.x_lengths = tf.placeholder(tf.int32,  shape=[self.batch_size,])
        self.x_lengths=tf.constant(self.n_steps,shape=[self.batch_size,])
        
        #self.imputed_deltapre=tf.placeholder(tf.float32,  shape=[self.batch_size,self.n_steps,self.n_inputs])
        self.imputed_delta=tf.constant(1.0,  shape=[self.batch_size,self.n_steps,self.n_inputs])
        self.imputed_m = tf.constant(1.0,  shape=[self.batch_size,self.n_steps,self.n_inputs])
        
        #self.ita = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs]) 
        self.ita = tf.placeholder(tf.float32, [None, self.z_dim]) 
        
        combined_x=tf.concat([self.aq_lastvalues,self.meo_lastvalues],2)
        combined_delta=tf.concat([self.aq_delta,self.meo_delta],2)
        combined_m=tf.concat([self.aq_m,self.meo_m],2)
        completed_x = tf.concat([self.complete_aq, self.complete_meo], 2)
        

        """ Loss Function """

        Pre_x = self.pretrainG(combined_x, combined_m, self.ita, combined_delta, self.x_lengths, self.keep_prob, is_training=True, reuse=False)
        # 每个序列length不一样，除的时候考虑只除length长度
        self.pretrain_loss = tf.reduce_sum(tf.square(tf.multiply(Pre_x, combined_m) -tf.multiply(combined_x, combined_m))) / tf.cast(tf.reduce_sum(combined_m), tf.float32)
        
        #discriminator(self, X,  DeltaPre, X_lengths,Keep_prob, is_training=True, reuse=False, isTdata=True):
        
        D_real, D_real_logits = self.discriminator(combined_x, combined_delta, self.x_lengths, self.keep_prob, is_training=True, reuse=False,isTdata=True)
        
        #generator(self,  originalX, originalM, z, Keep_prob, is_training=True, reuse=False):
        #G return total_result,self.imputed_deltapre,self.x_lengths
        g_x, g_deltapre, G_x_lengths  = self.generator(combined_x, combined_m, self.ita, combined_delta, self.x_lengths, self.keep_prob, is_training=True, reuse=True)
        
        D_fake, D_fake_logits = self.discriminator(g_x, g_deltapre, G_x_lengths,self.keep_prob,is_training=True, reuse=True ,isTdata=False)
        
        
        l2_loss = tf.reduce_sum(tf.square(tf.multiply(g_x, combined_m) - tf.multiply(combined_x, combined_m))) / tf.cast(tf.reduce_sum(combined_m),tf.float32)

        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)
        g_loss =  -d_loss_fake + self.g_loss_lambda * l2_loss 
        #d_loss = d_loss_real + d_loss_fake 
        d_loss = d_loss_real - g_loss 


        self.imputed_x = tf.multiply(combined_x, combined_m) + tf.multiply(g_x, (1 - combined_m))
        self.l2_loss = l2_loss
        self.d_loss_fake = d_loss_fake 
        self.g_x = g_x
        self.d_loss = d_loss
        self.g_loss = g_loss 

        self.mse = tf.reduce_sum(tf.square(tf.multiply(self.imputed_x, (1-combined_m)) - tf.multiply(completed_x, (1-combined_m)))) / tf.cast(tf.reduce_sum(1-combined_m),tf.float32)
        
        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        
        
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # this code have used batch normalization, so the upside line should be executed
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                        .minimize(self.d_loss, var_list=d_vars)
            #self.d_optim=self.optim(self.learning_rate, self.beta1,self.d_loss,d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                        .minimize(self.g_loss, var_list=g_vars)
            #self.g_optim=self.optim(self.learning_rate, self.beta1,self.g_loss,g_vars)
            self.g_pre_optim=tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                        .minimize(self.pretrain_loss,var_list=g_vars)
    
        
        print(0.01)
        self.clip_all_vals = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in t_vars]
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]
        self.clip_G = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in g_vars]
        # it's very useful to clip weights into [-0.1,0.1] , if you clip them into [-1,1], you will meet the exploding gradients.

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

    def pretrain(self, start_epoch,counter,start_time):
        
        if start_epoch < self.pretrain_epoch:
            #todo
            for epoch in range(start_epoch, self.pretrain_epoch):
            # get batch data
                idx=0
                #x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
                for now_aq,now_aq_m,now_aq_last,now_aq_delta,now_aq_complete,now_meo,now_meo_m,now_meo_last,now_meo_delta,now_meo_complete,now_time,_ in self.datasets.next_train_data(self.n_steps,self.batch_size):
                    
                    _ = self.sess.run(self.clip_all_vals)
                    #ita = np.random.normal(0, 0.01, size=(self.batch_size, self.n_steps, self.n_inputs))
                    ita = np.random.normal(0, 0.01, size=(self.batch_size, self.z_dim))
                    _, summary_str, p_loss = self.sess.run([self.g_pre_optim, self.g_pretrain_sum, self.pretrain_loss],
                                                   feed_dict={
                                                              self.aq: now_aq,
                                                              self.aq_m: now_aq_m,
                                                              self.aq_delta: now_aq_delta,
                                                              self.aq_lastvalues: now_aq_last,
                                                              self.meo: now_meo,
                                                              self.meo_m: now_meo_m,
                                                              self.meo_delta: now_meo_delta,
                                                              self.meo_lastvalues: now_meo_last,
                                                              self.ita: ita,
                                                              self.keep_prob: 0.8})
                    self.writer.add_summary(summary_str, counter)
    
    
                    counter += 1
    
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, pretrain_loss: %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, p_loss))
                    idx+=1
                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0



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
            return True
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
            
            idx=0
            for now_aq,now_aq_m,now_aq_last,now_aq_delta,now_aq_complete,now_meo,now_meo_m,now_meo_last,now_meo_delta,now_meo_complete,now_time,_ in self.datasets.next_train_data(self.n_steps,self.batch_size):

                #ita = np.random.normal(0, 0.01, size=(self.batch_size, self.n_steps, self.n_inputs))
                ita = np.random.normal(0, 0.01, size=(self.batch_size, self.z_dim))
                #_ = self.sess.run(self.clip_D)
                _ = self.sess.run(self.clip_all_vals)
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                               feed_dict={
                                                          self.aq: now_aq,
                                                          self.aq_m: now_aq_m,
                                                          self.aq_delta: now_aq_delta,
                                                          self.aq_lastvalues: now_aq_last,
                                                          self.meo: now_meo,
                                                          self.meo_m: now_meo_m,
                                                          self.meo_delta: now_meo_delta,
                                                          self.meo_lastvalues: now_meo_last,
                                                          self.ita: ita, 
                                                          self.keep_prob: 0.8})
                self.writer.add_summary(summary_str, counter)

                # update D network
                if counter%self.disc_iters==0:
                    #batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], 
                                                           feed_dict={
                                                           self.aq: now_aq,
                                                           self.aq_m: now_aq_m,
                                                           self.aq_delta: now_aq_delta,
                                                           self.aq_lastvalues: now_aq_last,
                                                           self.meo: now_meo,
                                                           self.meo_m: now_meo_m,
                                                           self.meo_delta: now_meo_delta,
                                                           self.meo_lastvalues: now_meo_last,
                                                           self.ita: ita,
                                                           self.keep_prob: 0.8})
                    self.writer.add_summary(summary_str, counter)
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f,counter:%4d" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss,counter))
                    #debug  

                counter += 1

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, counter:%4d" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, g_loss, counter))

                
                if np.mod(counter, 300) == 0 :
                    fake_x = self.sess.run([self.imputed_x],
                                            feed_dict={
                                                       self.aq: now_aq,
                                                       self.aq_m: now_aq_m,
                                                       self.aq_delta: now_aq_delta,
                                                       self.aq_lastvalues: now_aq_last,
                                                       self.meo: now_meo,
                                                       self.meo_m: now_meo_m,
                                                       self.meo_delta: now_meo_delta,
                                                       self.meo_lastvalues: now_meo_last,
                                                       self.ita: ita,
                                                       self.keep_prob: 0.8})
                    
                    self.writeG_Samples("G_sample_x",counter,fake_x)
                    
                idx+=1
            start_batch_id = 0
        
        self.save(self.checkpoint_dir, counter)
        
    def imputation(self,dataset,run_type):
        self.datasets=dataset
        if run_type == 1:
            start_time = time.time()
            batchid=1
            counter=1
            for now_aq,now_aq_m,now_aq_last,now_aq_delta,now_aq_complete,now_meo,now_meo_m,now_meo_last,now_meo_delta,now_meo_complete,now_time,_ in self.datasets.next_train_data(self.n_steps,self.batch_size):
                #ita = np.random.normal(0, 0.01, size=(self.batch_size, self.n_steps, self.n_inputs))
                ita = np.random.normal(0, 0.01, size=(self.batch_size, self.z_dim))
                gx, l2loss, p_fake, gloss = self.sess.run([self.imputed_x, self.l2_loss, self.d_loss_fake, self.g_loss],
                                                       feed_dict={
                                                                  self.aq: now_aq,
                                                                  self.complete_aq:now_aq_complete,
                                                                  self.aq_m: now_aq_m,
                                                                  self.aq_delta: now_aq_delta,
                                                                  self.aq_lastvalues: now_aq_last,
                                                                  self.meo: now_meo,
                                                                  self.complete_meo:now_meo_complete,
                                                                  self.meo_m: now_meo_m,
                                                                  self.meo_delta: now_meo_delta,
                                                                  self.meo_lastvalues: now_meo_last,
                                                                  self.ita:ita,
                                                                  self.keep_prob: 0.8})
                counter+=1
                print("Batchid: [%2d]  time: %4.4f, l2_loss: %.8f, p_fake: %.8f, gloss: %.8f"  % ( batchid, time.time() - start_time, l2loss, p_fake, gloss))
                self.save_imputation(gx, batchid, self.sess.run(self.x_lengths), self.sess.run(self.imputed_delta), now_time, run_type)
                batchid+=1
            return None

        if run_type == 2:
            start_time = time.time()
            batchid=1
            counter=1
            mses = []
            for now_aq,now_aq_m,now_aq_last,now_aq_delta,now_aq_complete, now_meo,now_meo_m,now_meo_last,now_meo_delta, now_meo_complete, now_time,_ in self.datasets.next_val_data(self.n_steps,self.batch_size):

                #ita = np.random.normal(0, 0.01, size=(self.batch_size, self.n_steps, self.n_inputs))
                ita = np.random.normal(0, 0.01, size=(self.batch_size, self.z_dim))
                gx, mse, p_fake, gloss = self.sess.run([self.imputed_x, self.mse, self.d_loss_fake, self.g_loss],
                                                       feed_dict={
                                                                  self.aq: now_aq,
                                                                  self.complete_aq:now_aq_complete,
                                                                  self.aq_m: now_aq_m,
                                                                  self.aq_delta: now_aq_delta,
                                                                  self.aq_lastvalues: now_aq_last,
                                                                  self.meo: now_meo,
                                                                  self.complete_meo:now_meo_complete,
                                                                  self.meo_m: now_meo_m,
                                                                  self.meo_delta: now_meo_delta,
                                                                  self.meo_lastvalues: now_meo_last,
                                                                  self.ita:ita,
                                                                  self.keep_prob: 0.8})
                counter+=1
                print("Batchid: [%2d]  time: %4.4f, mse: %.8f, p_fake: %.8f, gloss: %.8f"  % ( batchid, time.time() - start_time, mse, p_fake, gloss))
                mses.append(mse)
                self.save_imputation(gx, batchid, self.sess.run(self.x_lengths), self.sess.run(self.imputed_delta), now_time, run_type)
                batchid+=1
            print(sum(mses)/len(mses))
            return sum(mses)/len(mses)

        if run_type == 3:
            start_time = time.time()
            batchid=1
            counter=1
            mses = []
            for now_aq,now_aq_m,now_aq_last,now_aq_delta,now_aq_complete, now_meo,now_meo_m,now_meo_last,now_meo_delta, now_meo_complete, now_time,_ in self.datasets.next_test_data(self.n_steps,self.batch_size):

                #ita = np.random.normal(0, 0.01, size=(self.batch_size, self.n_steps, self.n_inputs))
                ita = np.random.normal(0, 0.01, size=(self.batch_size, self.z_dim))
                gx, mse, p_fake, gloss = self.sess.run([self.imputed_x, self.mse, self.d_loss_fake, self.g_loss],
                                                       feed_dict={
                                                                  self.aq: now_aq,
                                                                  self.complete_aq:now_aq_complete,
                                                                  self.aq_m: now_aq_m,
                                                                  self.aq_delta: now_aq_delta,
                                                                  self.aq_lastvalues: now_aq_last,
                                                                  self.meo: now_meo,
                                                                  self.complete_meo:now_meo_complete,
                                                                  self.meo_m: now_meo_m,
                                                                  self.meo_delta: now_meo_delta,
                                                                  self.meo_lastvalues: now_meo_last,
                                                                  self.ita:ita,
                                                                  self.keep_prob: 0.8})
                counter+=1
                print("Batchid: [%2d]  time: %4.4f, mse: %.8f, p_fake: %.8f, gloss: %.8f"  % ( batchid, time.time() - start_time, mse, p_fake, gloss))
                mses.append(mse)
                self.save_imputation(gx, batchid, self.sess.run(self.x_lengths), self.sess.run(self.imputed_delta), now_time, run_type)
                batchid+=1
            print(sum(mses)/len(mses))
            return sum(mses)/len(mses)
            #self.run_grui()

    def run_grui(self):
        random.seed()
        gpu=random.randint(0,1)
        command="CUDA_VISIBLE_DEVICES="+str(gpu)+" python ../GRUI_AQ/run_GAN_imputed_aq.py"+\
        ' --data-path=\"../Gan_Imputation/imputation_train_results/WGAN_AQ_'+ str(self.missing_rate) + '/'+self.model_dir+'\"' + ' --n-inputs=' + str(self.n_inputs) 
        os.system(command)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            self.epoch,self.disc_iters,self.n_inputs,
            self.batch_size, self.z_dim,
            self.lr,
            self.isNormal,self.isbatch_normal,
            self.n_steps,
            self.g_loss_lambda,self.beta1,self.n_hidden_units,
            self.pretrain_epoch, self.missing_rate
            )

    
    def save_imputation(self,impute_out, batchid, data_x_lengths, imputed_delta, now_time, run_type):
        #填充后的数据全是n_steps长度！，但只有data_x_lengths才是可用的
        if run_type == 1:
            imputation_dir="imputation_train_results"
        elif run_type == 2:
            imputation_dir="imputation_val_results"
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
        
        #write delta:list
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
        
        #write timesteps
        resultFile=open(os.path.join(imputation_dir,\
                                     self.model_name,\
                                     self.model_dir,\
                                     "batch"+str(batchid)+"time"),'w')
        for oneSeries in now_time:
            resultFile.writelines(str(oneSeries[0])+"\r\n")
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
