#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:54:53 2018

@author: yonghong luo
"""
from __future__ import print_function
import sys
sys.path.append("..")
import E2EGAN_PHY
import tensorflow as tf
import argparse
import numpy as np
from Physionet2012Data import readData, readTestValData
import os

"""main"""
def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gen-length', type=int, default=96)
    parser.add_argument('--impute-iter', type=int, default=400)
    parser.add_argument('--pretrain-epoch', type=int, default=5)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--data-path', type=str, default="../set-a/")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--g-loss-lambda',type=float,default=100)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--n-inputs', type=int, default=41)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--z-dim', type=int, default=64)
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
    parser.add_argument('--disc-iters',type=int,default=8)
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

    #make the max step length of two datasett the same
    #epochs=[30]
    g_loss_lambdas=[0.2,0.5,1,5,8,10,16,20,50]
    disc_iters = [1,2,3,4,5,6,7,8]
    epochs=[10]
    #g_loss_lambdas=[16]
    #disc_iters = [6]
    for disc in disc_iters:
        for e in epochs:
            for g_l in g_loss_lambdas:
                args.epoch=e
                args.disc_iters = disc
                args.g_loss_lambda=g_l
                tf.reset_default_graph()
                dt_train=readData.ReadPhysionetData(os.path.join(args.data_path,"train"), os.path.join(args.data_path,"train","list.txt"),isNormal=args.isNormal,isSlicing=args.isSlicing)
                dt_test=readTestValData.ReadPhysionetData(os.path.join(args.data_path,"test"), os.path.join(args.data_path,"test","list.txt"),dt_train.maxLength,isNormal=args.isNormal,isSlicing=args.isSlicing)
                dt_val=readTestValData.ReadPhysionetData(os.path.join(args.data_path,"val"), os.path.join(args.data_path,"val","list.txt"),dt_train.maxLength,isNormal=args.isNormal,isSlicing=args.isSlicing)
                tf.reset_default_graph()
                config = tf.ConfigProto() 
                config.gpu_options.allow_growth = True 
                with tf.Session(config=config) as sess:
                    gan = E2EGAN_PHY.E2EGAN(sess,
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
                
                    print(" [*] Train dataset Imputation begin!")
                    gan.imputation(dt_train, 1)
                    print(" [*] Train dataset Imputation finished!")
                

                    print(" [*] Val dataset Imputation begin!")
                    gan.imputation(dt_val, 2)
                    print(" [*] Val dataset Imputation finished!")

                    print(" [*] Test dataset Imputation begin!")
                    gan.imputation(dt_test, 3)
                    print(" [*] Test dataset Imputation finished!")
                tf.reset_default_graph()
if __name__ == '__main__':
    main()
