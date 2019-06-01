#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:23:18 2018

@author: yonghong
"""
import os 

def f():

    folders =  os.listdir("./checkpoint_physionet_imputed/")
    globalMax=0.0
    final_result = open("final_result", "w")
    globalf = ''
    for folder in folders:
        result_file = open("./checkpoint_physionet_imputed/" + folder + "/test_result")
        line = result_file.readlines()[0]
        final_result.write(folder + ": ")
        final_result.write(line)
        final_result.write("\n")
        final_result.flush()
        now_auc = float(line.split(":")[-1].strip())
        if now_auc > globalMax:
            globalMax = now_auc
            globalf = folder + ": " + line
    print(globalMax)
    print(globalf)
    final_result.close()

if __name__=="__main__":
    f()
            
        
