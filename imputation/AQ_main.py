#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:54:53 2018

@author: lyh
"""
from __future__ import print_function
import sys
sys.path.append("..")
import E2EGAN_AQ
import tensorflow as tf
import argparse
import numpy as np
from AirQualityData import readData_for_gan
import os

"""main"""
def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--pretrain-epoch', type=int, default=10)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--g-loss-lambda',type=float,default=0.0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--n-inputs', type=int, default=132)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--z-dim', type=int, default=128)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--isNormal',type=int,default=1)
    #0 false 1 true
    parser.add_argument('--isBatch-normal',type=int,default=1)
    parser.add_argument('--isSlicing',type=int,default=1)
    parser.add_argument('--disc-iters',type=int,default=1)
    parser.add_argument('--n-steps',type=int,default=48)
    parser.add_argument('--missing-rate',type=float,default=0.5)
    parser.add_argument('--beta1',type=float,default=0.98)
    parser.add_argument('--use-grui',type=int, default=1)
    


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
    if args.use_grui==1: 
            args.use_grui=True 
    #make the max step length of two datasett the same
    
    
    dt_train=readData_for_gan.ReadAriQualityDataForGan("2017-01-30 00:00:00", "2017-11-15 23:00:00", args.missing_rate, "2017-11-16 00:00:00", "2017-12-25 23:00:00", "2017-12-26 00:00:00", "2018-01-30 00:00:00",args.isNormal)
    

    min_mse = 5.0
    msesssssss = []
    min_paras = []
    disc_iters = [1,2,3,4,5,6,7,8,9]
    #lambdas = [0.1, 0.5, 1, 2, 5, 10,20,40]
    #disc_iters = [5]
    if args.missing_rate == 0.5:
        lambdas = [2]
    else :
        lambdas = [1,2,3,4,5,6,7,8,9]
    for disc in disc_iters:
        for lam in lambdas:
            args.disc_iters = disc
            args.g_loss_lambda = lam 

            tf.reset_default_graph()
            
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True 
            with tf.Session(config=config) as sess:
                gan = E2EGAN_AQ.E2EGAN_AQ(sess,
                            args=args,
                            datasets=dt_train,
                            )

                # build graph
                gan.build_model()

                # show network architecture
                #show_all_variables()

                # launch the graph in a session
                gan.train()
                print(" [*] Training finished!")

                print(" [*] Begin training set imputation!")
                gan.imputation(dt_train,1)
                print(" [*] Training dataset Imputation finished!")
                
                print(" [*] Begin validation set imputation!")
                now_mse = gan.imputation(dt_train,2)
                msesssssss.append(now_mse)
                print(" [*] Validation dataset Imputation finished!")
                if now_mse < min_mse:
                    min_mse = now_mse
                    min_paras = []
                    min_paras.append(disc)
                    min_paras.append(lam)

                #print(" [*] Begin training set imputation!")
                #gan.imputation(dt_train,3)
                #print(" [*] Training dataset Imputation finished!")
            tf.reset_default_graph()
    

    print("min val mse:%.8f"%min_mse)
    print("min val paras are disc: %.2f, lambda: %.2f "%(min_paras[0], min_paras[1]) )

    args.disc_iters = min_paras[0]
    args.g_loss_lambda = min_paras[1]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True 
    with tf.Session(config=config) as sess:
        gan = E2EGAN_AQ.E2EGAN_AQ(sess,
                    args=args,
                    datasets=dt_train,
                    )

        # build graph
        gan.build_model()

        # show network architecture
        #show_all_variables()

        # launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        print(" [*] Begin testing set imputation!")
        mse = gan.imputation(dt_train,3)
        print(" [*] Testing dataset Imputation finished!")
        print("[*] Testing dataset Imputation mse is: %.8f " %(mse))


    print("val mse are:")
    print(msesssssss)
    

if __name__ == '__main__':
    main()
