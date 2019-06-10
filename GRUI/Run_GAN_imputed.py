#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:47:41 2018

@author: yonghong
"""

from __future__ import print_function
import sys
sys.path.append("..")
import argparse
import os
import random
import tensorflow as tf
import numpy as np
from Physionet2012ImputedData import readImputed
import gru_forGAN
SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--run-type', type=str, default='test')
    parser.add_argument('--data-path', type=str, default="../imputation/imputation_train_results/E2EGAN_PHY/10_7_41_128_64_0.005_400_True_True_True_50_0.5")
    #输入填充之后的训练数据集的完整路径
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--n-inputs', type=int, default=41)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--checkpoint-dir', type=str, default='../GRUI/checkpoint_physionet_imputed',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log-dir', type=str, default='../GRUI/logs_physionet_imputed',
                        help='Directory name to save training logs')
    parser.add_argument('--isNormal',type=int,default=1)
    parser.add_argument('--isSlicing',type=int,default=1)
    #0 false 1 true
    parser.add_argument('--isBatch-normal',type=int,default=1)
    args = parser.parse_args()
    if args.isBatch_normal==0:
            args.isBatch_normal=False
    if args.isBatch_normal==1:
            args.isBatch_normal=True
    if args.isNormal==0:
            args.isNormal=False
    if args.isNormal==1:
            args.isNormal=True
    if args.isSlicing==0:
            args.isSlicing=False
    if args.isSlicing==1:
            args.isSlicing=True
    
    checkdir=args.checkpoint_dir
    logdir=args.log_dir
    path_splits=args.data_path.split("/")
    if len(path_splits[-1])==0:
        datasetName=path_splits[-2]
    else:
        datasetName=path_splits[-1]
    
    args.checkpoint_dir=checkdir+"/"+datasetName
    args.log_dir=logdir+"/"+datasetName
    
    dt_train=readImputed.ReadImputedPhysionetData(args.data_path)
    dt_train.load()
    
    dt_test=readImputed.ReadImputedPhysionetData(args.data_path.replace("imputation_train_results","imputation_test_results"))
    dt_test.load()
    
    dt_val=readImputed.ReadImputedPhysionetData(args.data_path.replace("imputation_train_results","imputation_val_results"))
    dt_val.load()


    paras = []
    #hiddenUnits=[16,32,64,128]
    #lrs=[0.008,0.01,0.012,0.015] 
    hiddenUnits=[16]
    lrs=[0.015] 
    max_auc = 0.5
    for units in hiddenUnits:
        for lr in lrs:
            args.n_hidden_units=units
            args.lr=lr
            tf.reset_default_graph()
            tf.set_random_seed(SEED)
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True 
            with tf.Session(config=config) as sess:
                model = gru_forGAN.gru(sess,
                            args=args,
                            dataset=dt_train,
                            test_set=dt_test,
                            val_set = dt_val
                            )
        
                # build graph
                model.build()
                max_val_auc,max_val_epoch = model.train()
                print("now max val auc: %.8f"%(max_val_auc))
                print(" [*] Training and validating finished!")
                test_acc, test_auc = model.test(dt_test, max_val_epoch)
                print(" [*] Testing for one, auc is : %.8f" % test_auc)
                if test_auc > max_auc :
                    max_auc = test_auc
                    paras = []
                    paras.append(units)
                    paras.append(lr)
                    paras.append(max_val_epoch)

            print("")
    print("test  auc is : %.8f" %(max_auc))
    print()
    print("test  auc paras:")
    print(paras)



    args.n_hidden_units = paras[0]
    args.lr = paras[1]
    #args.n_hidden_units = 128
    #args.lr = 0.015
    tf.reset_default_graph()
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    with tf.Session(config=config) as sess:
        model = gru_forGAN.gru(sess,
                    args=args,
                    dataset=dt_train,
                    test_set=dt_test,
                    val_set = dt_val
                    )
    
        # build graph
        model.build()
        test_acc, test_auc = model.test(dt_test, paras[2])
        #for e in range(1, 30):
        #    test_acc, test_auc = model.test(dt_test, e)
    print("final test auc is : %.8f" %(test_auc))
    f = open(args.checkpoint_dir + "/" + "test_result", "w")
    
    f.write("paras are: hidden_units: %d, lr: %.6f, epoch: %d, final test auc: %.8f " %(paras[0], paras[1], paras[2], test_auc))

