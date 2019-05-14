# -*- coding: utf-8 -*-
import random
import os
import time
import math
import sys
import numpy as np

station_length=11
def get_data_list(file_name):

    with open(file_name, 'r') as f:
        data_list=[]
        last_line=''
        for line in f:
            line=(line.replace("\r","").strip('\n')).split(',') 
            if last_line!=line:
                data_list.append(line)
                last_line=line
    return data_list


def list_to_file(list, file_name):
    """
    list - file
    
    :return: None
    """
    with open(file_name, 'w') as f:
        for line in list:
            f.write(str(line).replace("\r",""))
            f.write('\n')
            f.flush()
    return None
#['aotizhongxin_aq', '2017-03-03 00:00:00', '70.0', '106.0', '83.0', '1.7', '26.0', '22.0']
#['shunyi_meo', '116.6152777777778', '40.126666666666665', '2017-03-03 00:00:00', '2.6', '1010.2', '43', '28.0', '1.8', 'Sunny/clear']

def file_to_list(file_name):
    """
    format:['aotizhongxin_aq', '2017-03-03 00:00:00', '70.0', '106.0', '83.0', '1.7', '26.0', '22.0']
           ['shunyi_meo', '116.6152777777778', '40.126666666666665', '2017-03-03 00:00:00', '2.6', '1010.2', '43', '28.0', '1.8', 'Sunny/clear']
  
    :return: list
    """
    with open(file_name, 'r') as f:
        data_list=[]
        for line in f:
            line=line.strip()
            line=line.replace("]","")
            line=line.replace("[","")
            line=line.replace(" '","")
            line=line.replace("'","")
            data_list.append(line.split(','))
    return data_list


def count_print(data_list):
    """
    :return: None
    """
    attributes_name_list = data_list[0]
    attributes_num = len(attributes_name_list)
    print('属性名称列表：',attributes_name_list)
    attributes_dict = {}
    for i in range(attributes_num):
        attributes_name = attributes_name_list[i]
        attributes_dict[attributes_name] = set([line[i] for line in data_list[1:]])
        values_num = len(attributes_dict[attributes_name])
        print('属性名：%s 取值个数：%d 取值集合：%s'%(attributes_name, values_num, attributes_dict[attributes_name]))
    return None


def get_train_and_test(total_data, train_date1='2017-03-01 00:00:00', train_date2='2017-03-03 23:00:00', val_date1='2017-03-04 00:00:00', val_date2='2017-03-04 06:00:00', test_date1='2017-03-05 00:00:00', test_date2='2017-03-05 06:00:00'):
    """
    获取训练集和测试集
    :param total_data: 所有日期的数据(list)
    :param train_date1: 训练集选取的日期1
    :param train_date2: 训练集选取的日期2
    :param test_date: 测试集选取的日期
    :return: (训练集，测试集)(list)
    """
    utc_time_index = total_data[0].index('utc_time')
    train_data = []
    test_data = []
    val_data = []
    for line in total_data:
        if (line[utc_time_index] >= train_date1) and (line[utc_time_index] <= train_date2):
            train_data.append(line)
        if (line[utc_time_index] >= val_date1) and (line[utc_time_index] <= val_date2):
            val_data.append(line)
        if (line[utc_time_index] >= test_date1) and (line[utc_time_index] <= test_date2):
            test_data.append(line)
    return train_data, val_data, test_data


def randomly_discard_data(total_data, p, attri_num):
    """
    随机剪掉一些数据
    :param total_data: 未剪过的数据（list）
    :param p: 丢弃的属性值的比例,missing rate, 原始数据missing rate为0.141
    :param attri_num: 待剪属性的个数
    :return: 剪后的数据(list), 被剪数据的坐标
    """
    remain_data = []
    for line in total_data:
        tmp = []
        for item in line:
            tmp.append(item)
        remain_data.append(tmp)

    total_no_missing = len(total_data) * attri_num * (1 - 0.141)
    discard_num = int(total_no_missing * p)
    k = 0
    random.seed()
    while k < discard_num:
        random_point = [random.randint(0, len(total_data)-1), random.randint(-attri_num, -1)]
        if remain_data[random_point[0]][random_point[1]] != '':
            remain_data[random_point[0]][random_point[1]] = ''
            k = k + 1
    return remain_data

def readImputedX(x):
    this_x=[]
    this_lengths=[]
    count=1
    for line in x.readlines():
        if count==1:
            words=line.strip().split(",")
            for w in words:
                if w=='':
                    continue
                this_lengths.append(int(w))
        else:
            if "end" in line:
                continue
            if "begin" in line:
                d=[]
                this_x.append(d)
            else:
                words=line.strip().split(",")
                oneclass=[]
                for w in words:
                    if w=='':
                        continue
                    oneclass.append(float(w))
                this_x[-1].append(oneclass)
        count+=1
    return this_x,this_lengths    
    
def readImputedDelta(delta):
    this_delta=[]
    for line in delta.readlines():
        if "end" in line:
            continue
        if "begin" in line:
            d=[]
            this_delta.append(d)
        else:
            words=line.strip().split(",")
            oneclass=[]
            for i in range(len(words)):
                w=words[i]
                if w=='':
                    continue
                oneclass.append(float(w))
            this_delta[-1].append(oneclass)
    return this_delta

def readTime(file_time):
    this_time=[]
    for line in file_time.readlines():
        this_time.append(float(line.strip()))
            
    return this_time
        
    
    
class ReadAriQualityDataForGan():
    #2017.7.2-7.8缺失了很多天的空气质量数据！
    #有bug,训练集的结束时间不能大于测试集的开始时间
    def __init__(self, train_date_1, train_date_2, p, val_date1, val_date2, test_date1, test_date2, isNormal):
        self.train_date_1=train_date_1
        self.train_date_2=train_date_2
        self.test_date_1=test_date1
        self.test_date_2=test_date2
        self.val_date_1=val_date1 
        self.val_date_2=val_date2
        self.station_length=11
        self.station_dic=stations()
        self.wea_dic=weathers()
        processed_path=train_date_1+"_"+str(p)
        
        if os.path.exists(os.path.join("../AirQualityData",processed_path,"aq_train"+str(p)+".txt")):
            remain_aq_train=file_to_list("../AirQualityData/"+processed_path+'/aq_train'+str(p)+'.txt')

            remain_aq_test=file_to_list("../AirQualityData/"+processed_path+'/aq_test'+str(p)+'.txt')

            remain_aq_val=file_to_list("../AirQualityData/"+processed_path+'/aq_val'+str(p)+'.txt')

            remain_meo_train=file_to_list("../AirQualityData/"+processed_path+'/meo_train'+str(p)+'.txt')

            remain_meo_test=file_to_list("../AirQualityData/"+processed_path+'/meo_test'+str(p)+'.txt')

            remain_meo_val=file_to_list("../AirQualityData/"+processed_path+'/meo_val'+str(p)+'.txt')

            complete_aq_train=file_to_list("../AirQualityData/"+processed_path+'/aq_train.txt')

            complete_aq_test=file_to_list("../AirQualityData/"+processed_path+'/aq_test.txt')

            complete_aq_val=file_to_list("../AirQualityData/"+processed_path+'/aq_val.txt')

            complete_meo_train=file_to_list("../AirQualityData/"+processed_path+'/meo_train.txt')

            complete_meo_test=file_to_list("../AirQualityData/"+processed_path+'/meo_test.txt')

            complete_meo_val=file_to_list("../AirQualityData/"+processed_path+'/meo_val.txt')
        else:
            # not exist
            if not os.path.exists("../AirQualityData/"+processed_path):
                os.makedirs("../AirQualityData/"+processed_path)
            aq_data_list = get_data_list('../AirQualityData/originData/beijing_17_18_aq.csv')
            meo_data_list = get_data_list('../AirQualityData/originData/beijing_17_18_meo.csv')
            # 抽取三天的数据作为训练集和测试集
            aq_train_data, aq_val_data, aq_test_data = get_train_and_test(aq_data_list, train_date_1, train_date_2, val_date1, val_date2, test_date1, test_date2)
            meo_train_data, meo_val_data, meo_test_data = get_train_and_test(meo_data_list, train_date_1, train_date_2, val_date1, val_date2, test_date1, test_date2)
            
            list_to_file(aq_train_data, "../AirQualityData/"+processed_path+'/aq_train.txt')
            list_to_file(aq_test_data, "../AirQualityData/"+processed_path+'/aq_test.txt')
            list_to_file(aq_val_data, "../AirQualityData/"+processed_path+'/aq_val.txt')
            list_to_file(meo_train_data, "../AirQualityData/"+processed_path+'/meo_train.txt')
            list_to_file(meo_test_data, "../AirQualityData/"+processed_path+'/meo_test.txt')
            list_to_file(meo_val_data, "../AirQualityData/"+processed_path+'/meo_val.txt')
            # complete datasets
            
            remain_aq_train  = randomly_discard_data(aq_train_data, p, 6 )
            remain_aq_test  = randomly_discard_data(aq_test_data, p, 6)
            remain_aq_val  = randomly_discard_data(aq_val_data, p, 6)
            remain_meo_train  = randomly_discard_data(meo_train_data, p, 6)
            remain_meo_test = randomly_discard_data(meo_test_data, p, 6)  
            remain_meo_val = randomly_discard_data(meo_val_data, p, 6)  
            
            list_to_file(remain_aq_train, "../AirQualityData/"+processed_path+'/aq_train'+str(p)+'.txt')
            list_to_file(remain_aq_test, "../AirQualityData/"+processed_path+'/aq_test'+str(p)+'.txt')
            list_to_file(remain_aq_val, "../AirQualityData/"+processed_path+'/aq_val'+str(p)+'.txt')
            list_to_file(remain_meo_train, "../AirQualityData/"+processed_path+'/meo_train'+str(p)+'.txt')
            list_to_file(remain_meo_test, "../AirQualityData/"+processed_path+'/meo_test'+str(p)+'.txt')
            list_to_file(remain_meo_val, "../AirQualityData/"+processed_path+'/meo_val'+str(p)+'.txt')
            #incomplete datasets
            complete_aq_train = aq_train_data
            complete_aq_test = aq_test_data
            complete_aq_val = aq_val_data 

            complete_meo_train = meo_train_data
            complete_meo_test = meo_test_data
            complete_meo_val = meo_val_data
            
        self.aq_train_data,self.aq_train_m,self.times_train,self.aq_train_last,self.aq_train_delta=split_and_combine(remain_aq_train,train_date_1,train_date_2,self.wea_dic,self.station_dic)

        self.aq_test_data,self.aq_test_m,self.times_test,self.aq_test_last,    self.aq_test_delta =split_and_combine(remain_aq_test,test_date1,test_date2,self.wea_dic,self.station_dic)

        self.aq_val_data,self.aq_val_m,self.times_val,self.aq_val_last,    self.aq_val_delta =split_and_combine(remain_aq_val,val_date1,val_date2,self.wea_dic,self.station_dic)

        self.meo_train_data,self.meo_train_m,_          ,self.meo_train_last,self.meo_train_delta =split_and_combine(remain_meo_train,train_date_1,train_date_2,self.wea_dic,self.station_dic)

        self.meo_test_data,self.meo_test_m,_            ,self.meo_test_last,  self.meo_test_delta =split_and_combine(remain_meo_test,test_date1,test_date2,self.wea_dic,self.station_dic)

        self.meo_val_data,self.meo_val_m,_            ,self.meo_val_last,  self.meo_val_delta =split_and_combine(remain_meo_val,val_date1,val_date2,self.wea_dic,self.station_dic)

        self.complete_aq_data,_,_,_,_=split_and_combine(complete_aq_train,train_date_1,train_date_2,self.wea_dic,self.station_dic)

        self.complete_aq_data_test,_,_,_,_=split_and_combine(complete_aq_test,test_date1,test_date2,self.wea_dic,self.station_dic)
        
        self.complete_aq_data_val,_,_,_,_=split_and_combine(complete_aq_val,val_date1,val_date2,self.wea_dic,self.station_dic)

        self.complete_meo_data,_,_,_,_=split_and_combine(complete_meo_train,train_date_1,train_date_2,self.wea_dic,self.station_dic)

        self.complete_meo_data_test,_,_,_,_=split_and_combine(complete_meo_test,test_date1,test_date2,self.wea_dic,self.station_dic)
        
        self.complete_meo_data_val,_,_,_,_=split_and_combine(complete_meo_val,val_date1,val_date2,self.wea_dic,self.station_dic)


        if isNormal:

            normalization(self.aq_train_data,self.aq_train_last,self.aq_test_data,self.aq_test_last,self.aq_val_data,self.aq_val_last)

            normalization(self.meo_train_data,self.meo_train_last,self.meo_test_data,self.meo_test_last, self.meo_val_data,self.meo_val_last)

            normalization(self.complete_aq_data,None,self.complete_aq_data_test,None,self.complete_aq_data_val, None)

            normalization(self.complete_meo_data,None,self.complete_meo_data_test,None, self.complete_meo_data_val,None)
    
    def next_train_data(self,hours,batch_size):
        #meo and aq
        #first split by hours, then shuffle
        t1=get_seconds(self.train_date_1)
        t2=get_seconds(self.train_date_2)
        
        #index代表一共可以将一年数据分割成多少份，每份的大小是hours小时
        index=0
        while t1+hours*index*3600<=t2:
            index+=1
        index-=1
        used=[False]*index
        #count_代表已经用了多少份    
        count=0
        now_batch_size=0
        #now_batch_size代表已经选出了多少份/batch_size
        
        now_aq=[]
        now_aq_m=[]
        now_meo_m=[]
        now_meo=[]
        now_time=[]
        now_aq_last=[]
        now_meo_last=[]
        now_aq_delta=[]
        now_meo_delta=[]
        now_aq_complete=[]
        now_meo_complete=[]
        now_y=[]
        random.seed()
        while count<index:
            k=random.randint(0,index-1)
            #print(k)
            if not used[k]:
                used[k]=True
                count+=1
                now_batch_size+=1
                now_aq.append(self.aq_train_data[k*hours:(k+1)*hours])
                now_meo.append(self.meo_train_data[k*hours:(k+1)*hours])
                now_time.append(self.times_train[k*hours:(k+1)*hours])
                now_aq_m.append(self.aq_train_m[k*hours:(k+1)*hours])
                now_meo_m.append(self.meo_train_m[k*hours:(k+1)*hours])
                now_aq_last.append(self.aq_train_last[k*hours:(k+1)*hours])
                now_meo_last.append(self.meo_train_last[k*hours:(k+1)*hours])
                now_aq_delta.append(self.aq_train_delta[k*hours:(k+1)*hours])
                now_meo_delta.append(self.meo_train_delta[k*hours:(k+1)*hours])
                now_aq_complete.append(self.complete_aq_data[k*hours:(k+1)*hours])
                now_meo_complete.append(self.complete_meo_data[k*hours:(k+1)*hours])
                now_y.append([var[0:66] for var in self.complete_aq_data[(k+1)*hours:(k+1)*hours+6]])
            if now_batch_size==batch_size:
                yield now_aq,now_aq_m,now_aq_last,now_aq_delta,now_aq_complete, now_meo,now_meo_m,now_meo_last,now_meo_delta,now_meo_complete, now_time,now_y
                now_batch_size=0
                now_aq=[]
                now_meo=[]
                now_time=[]
                now_aq_m=[]
                now_meo_m=[]
                now_aq_last=[]
                now_meo_last=[]
                now_aq_delta=[]
                now_meo_delta=[]
                now_meo_complete = []
                now_aq_complete = []
                now_y=[]

    def next_test_data(self,hours,batch_size):
        # only meo
        t1=get_seconds(self.test_date_1)
        t2=get_seconds(self.test_date_2)
        
        #index代表一共可以将一年数据分割成多少份，每份的大小是batch_size*hours小时
        index=0
        while t1+hours*index*3600*batch_size<=t2:
            index+=1
        index-=1
    
        now_aq=[]
        now_aq_m=[]
        now_meo_m=[]
        now_meo=[]
        now_time=[]
        now_aq_last=[]
        now_meo_last=[]
        now_aq_delta=[]
        now_meo_delta=[]
        now_aq_complete=[]
        now_meo_complete=[]
        now_y=[]
        k=0
        while k<index:
            for i in range(1,batch_size+1):
                begin=k*hours*batch_size+(i-1)*hours
                end  =k*hours*batch_size+i*hours
                now_aq.append(self.aq_test_data[begin:end])
                now_meo.append(self.meo_test_data[begin:end])
                now_time.append(self.times_test[begin:end])
                now_aq_m.append(self.aq_test_m[begin:end])
                now_meo_m.append(self.meo_test_m[begin:end])
                now_aq_last.append(self.aq_test_last[begin:end])
                now_meo_last.append(self.meo_test_last[begin:end])
                now_aq_delta.append(self.aq_test_delta[begin:end])
                now_meo_delta.append(self.meo_test_delta[begin:end])
                now_aq_complete.append(self.complete_aq_data_test[begin:end])
                now_meo_complete.append(self.complete_meo_data_test[begin:end])
                now_y.append([var[0:66] for var in self.complete_aq_data_test[end:end+6]])
            yield now_aq,now_aq_m,now_aq_last,now_aq_delta,now_aq_complete, now_meo,now_meo_m,now_meo_last,now_meo_delta, now_meo_complete, now_time,now_y
            k+=1
            now_aq=[]
            now_meo=[]
            now_time=[]
            now_aq_m=[]
            now_meo_m=[]
            now_aq_last=[]
            now_meo_last=[]
            now_aq_delta=[]
            now_meo_delta=[]
            now_meo_complete = []
            now_aq_complete = []
            now_y=[]

    def next_val_data(self,hours,batch_size):
        # only meo
        t1=get_seconds(self.val_date_1)
        t2=get_seconds(self.val_date_2)
        
        #index代表一共可以将一年数据分割成多少份，每份的大小是batch_size*hours小时
        index=0
        while t1+hours*index*3600*batch_size<=t2:
            index+=1
        index-=1
    
        now_aq=[]
        now_aq_m=[]
        now_meo_m=[]
        now_meo=[]
        now_time=[]
        now_aq_last=[]
        now_meo_last=[]
        now_aq_delta=[]
        now_meo_delta=[]
        now_aq_complete=[]
        now_meo_complete=[]
        now_y=[]
        k=0
        while k<index:
            for i in range(1,batch_size+1):
                begin=k*hours*batch_size+(i-1)*hours
                end  =k*hours*batch_size+i*hours
                now_aq.append(self.aq_val_data[begin:end])
                now_meo.append(self.meo_val_data[begin:end])
                now_time.append(self.times_val[begin:end])
                now_aq_m.append(self.aq_val_m[begin:end])
                now_meo_m.append(self.meo_val_m[begin:end])
                now_aq_last.append(self.aq_val_last[begin:end])
                now_meo_last.append(self.meo_val_last[begin:end])
                now_aq_delta.append(self.aq_val_delta[begin:end])
                now_meo_delta.append(self.meo_val_delta[begin:end])
                now_aq_complete.append(self.complete_aq_data_val[begin:end])
                now_meo_complete.append(self.complete_meo_data_val[begin:end])
                now_y.append([var[0:66] for var in self.complete_aq_data_val[end:end+6]])
            yield now_aq,now_aq_m,now_aq_last,now_aq_delta,now_aq_complete, now_meo,now_meo_m,now_meo_last,now_meo_delta, now_meo_complete, now_time,now_y
            k+=1
            now_aq=[]
            now_meo=[]
            now_time=[]
            now_aq_m=[]
            now_meo_m=[]
            now_aq_last=[]
            now_meo_last=[]
            now_aq_delta=[]
            now_meo_delta=[]
            now_meo_complete = []
            now_aq_complete = []
            now_y=[]
            
def normalization(train,train_last,test,test_last,val,val_last):
    mean=[0.0]*len(train[0])
    count=[0.0]*len(train[0])
    std=[0.0]*len(train[0])
    for line in train:
        for i in range(len(line)):
            mean[i]+=line[i]
            if line[i]!=0:
                count[i]+=1
    for line in test:
        for i in range(len(line)):
            mean[i]+=line[i]
            if line[i]!=0:
                count[i]+=1

    for line in val:
        for i in range(len(line)):
            mean[i]+=line[i]
            if line[i]!=0:
                count[i]+=1
                
    for i in range(len(mean)):
        if count[i]!=0:
            mean[i]=mean[i]/count[i]
    
    for line in train:
        for i in range(len(line)):
            if line[i]!=0:
                std[i]+=(line[i]-mean[i])**2
                
    for line in test:
        for i in range(len(line)):
            if line[i]!=0:
                std[i]+=(line[i]-mean[i])**2

    for line in val:
        for i in range(len(line)):
            if line[i]!=0:
                std[i]+=(line[i]-mean[i])**2

    for i in range(len(mean)):
        if count[i]!=1:
            std[i]=math.sqrt(1.0/(count[i]-1)*std[i])
        
    for line in train:
        for i in range(len(line)):
            if line[i] !=0:
                if std[i]==0:
                    line[i]=0.0
                else:
                    line[i]=1.0/std[i]*(line[i]-mean[i]) 
                    
    for line in test:
        for i in range(len(line)):
            if line[i] !=0:
                if std[i]==0:
                    line[i]=0.0
                else:
                    line[i]=1.0/std[i]*(line[i]-mean[i]) 

    for line in val:
        for i in range(len(line)):
            if line[i] !=0:
                if std[i]==0:
                    line[i]=0.0
                else:
                    line[i]=1.0/std[i]*(line[i]-mean[i])

    if not train_last:
        pass
    else:
        for line in train_last:
            for i in range(len(line)):
                if line[i] !=0:
                    if std[i]==0:
                        line[i]=0.0
                    else:
                        line[i]=1.0/std[i]*(line[i]-mean[i]) 
    if not test_last:
        pass
    else:
        for line in test_last:
            for i in range(len(line)):
                if line[i] !=0:
                    if std[i]==0:
                        line[i]=0.0
                    else:
                        line[i]=1.0/std[i]*(line[i]-mean[i])


    if not val_last:
        pass
    else:
        for line in val_last:
            for i in range(len(line)):
                if line[i] !=0:
                    if std[i]==0:
                        line[i]=0.0
                    else:
                        line[i]=1.0/std[i]*(line[i]-mean[i]) 


            
def split_and_combine( data, begin_time, end_time,wea_dic,station_dic):
    #begin:2017-03-01 00:00:00 end:2017-03-03 23:00:00
    #返回所有的时间，所有的观测站点横向累加以及填充之后的结果
    splited_data=[[] for i in range(station_length)]
    
    for line in data:
        station_name=line[0].split("_")[0]
        if line[-1] in wea_dic:
            line[-1]=wea_dic[line[-1]]
        if station_name in station_dic:
            splited_data[station_dic[station_name]].append(line)
    
    combined_data=[]
    
    t1=get_seconds(begin_time)
    t2=get_seconds(end_time)
    
    steps=(t2-t1)/3600+1
    times=[]
    index=[0]*station_length
    
    for i in range(int(steps)):
        temp_time=t1+i*3600
        com=[]
        for j in range(station_length):
            if index[j]<len(splited_data[j]):
                line=splited_data[j][index[j]]
                if get_seconds(line[-7])==temp_time:
                    com.extend(line[-6:])
                    index[j]+=1
                else:
                    com.extend([0.0]*6)
            else:
                com.extend([0.0]*6)
        combined_data.append(com)
        times.append(temp_time)
    
    
    combined_m=[]
    for i in range(len(combined_data)):
        one_m=[0.0]*len(combined_data[i])
        for j in range(len(combined_data[i])):
            if combined_data[i][j] !='' and combined_data[i][j] !=0.0:
                combined_data[i][j]=float(combined_data[i][j])
                one_m[j]=1.0
            else:
                combined_data[i][j]=0.0
        combined_m.append(one_m)
    
    last_values=[]
    delta=[]
    for i in range(len(combined_data)):
        one_last=[0.0]*len(combined_data[i])
        last_values.append(one_last)
        
        one_delta=[0.0]*len(combined_data[i])
        delta.append(one_delta)
        
        for j in range(len(combined_data[i])):
            if i==0:
                last_values[i][j]=0.0 if combined_m[i][j]==0 else combined_data[i][j]
                continue
            if combined_m[i][j]==1:
                last_values[i][j]=combined_data[i][j]
            else:
                last_values[i][j]=last_values[i-1][j]
                
        
            if combined_m[i-1][j]==1:
                delta[i][j]=(times[i]-times[i-1])/3600
            else:
                delta[i][j]=(times[i]-times[i-1])/3600+delta[i-1][j]
        
    
    return combined_data,combined_m,times,last_values,delta
    
def timestamp_to_date(timestamp):
    time_local = time.localtime(timestamp)
    #转换成新的时间格式(2016-05-05 20:28:54)
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    return dt

def get_seconds(str_time):
    timeArray = time.strptime(str_time, "%Y-%m-%d %H:%M:%S")
    t=time.mktime(timeArray)
    return t
    
def weathers():
    dic_wea={"Sunny/clear":0,
             "Haze":1,
             "Snow":2,
             "Fog":3,
             "Rain":4,
             "Rain/Snow with Hail":5,
             "Rain with Hail":6,
             "Dust":7,
             "Sand":8,
             "Sleet":9
            }
    return dic_wea
	
def stations():
    dic_aq={"shunyi":0,
            "yanqing":1,
            "yanqin":1,
            "miyun":2,
            "huairou":3,
            "pinggu":4,
            "tongzhou":5,
            "pingchang":6,
            "mentougou":7,
            "fengtaihuayuan":8,
            "fengtai":8,
            "daxing":9,
            "fangshan":10,
            }
    
    return dic_aq


if __name__ == '__main__':
    """
    aq_data_list = get_data_list('originData/beijing_17_18_aq.csv')
    meo_data_list = get_data_list('originData/beijing_17_18_meo.csv')
    # 统计信息
    # print('空气质量数据统计：')
    # count_print(aq_data_list)
    # print()
    # print('气象数据统计：')
    # count_print(meo_data_list)

    # 抽取三天的数据作为训练集和测试集
    aq_train_data, aq_test_data = get_train_and_test(aq_data_list)
    meo_train_data, meo_test_data = get_train_and_test(meo_data_list)


    # 将完整数据写入磁盘
    list_to_file(aq_train_data, 'data/aq_train.txt')
    list_to_file(aq_test_data, 'data/aq_test.txt')
    list_to_file(meo_train_data, 'data/meo_train.txt')
    list_to_file(meo_test_data, 'data/meo_test.txt')

    # 随机剪掉一些数据
    remain_aq_train, aq_train_discard_points = randomly_discard_data(aq_train_data, 0.1, 6)
    remain_aq_test, aq_test_discard_points = randomly_discard_data(aq_test_data, 0.1, 6)
    remain_meo_train, meo_train_discard_points = randomly_discard_data(meo_train_data, 0.1, 6)
    remain_meo_test, meo_test_discard_points = randomly_discard_data(meo_test_data, 0.1, 6)

    # 将剪掉一些数据后的数据写入磁盘
    list_to_file(remain_aq_train, 'data/remain_aq_train.txt')
    list_to_file(remain_aq_test, 'data/remain_aq_test.txt')
    list_to_file(remain_meo_train, 'data/remain_meo_train.txt')
    list_to_file(remain_meo_test, 'data/remain_meo_test.txt')
    """
    
    
    a=ReadAriQualityDataForGan("2017-01-30 00:00:00", "2017-11-29 23:00:00", 0.5, "2017-11-30 00:00:00", "2018-12-30 23:00:00", "2017-12-31 00:00:00", "2018-01-30 00:00:00",False)
    now_timessss=[]
    count = 0
    count2 = 0
    index = 0
    for now_aq,now_aq_m,now_aq_last,now_aq_delta,now_aq_complete, now_meo,now_meo_m,now_meo_last,now_meo_delta,now_meo_complete, now_time,now_y in a.next_val_data(48,16):
        #print(now_time[0][0])
        #now_timessss.append(now_time)
        if index == 1:
            print("now_aq")
            print(now_aq[0][10])
            print("------------------------------")
            print("now_aq_m")
            print(now_aq_m[0][10])
            print("------------------------------")
            print("now_aq_last")
            print(now_aq_last[0][9])
            print(now_aq_last[0][10])
            print("------------------------------")
            print("now_aq_complete")
            print(now_aq_complete[0][10])
            print("------------------------------")
            print("now_aq_delta")
            print(now_aq_delta[0][10])

        index += 1
    
    
    # print(a.complete_aq_data[48])
    """
    basePath="../Gan_Imputation/aq_imputation_train_results/WGAN_AQ_0.5"+"/30_4_16_256_0.001_400_True_True_48_0.5_0.0_0.9/"
    a=ReadImputedAQData("2017-01-30 00:00:00", "2017-12-24 23:00:00", 0.5, "2017-12-25 00:00:00", "2018-01-30 00:00:00",False,basePath)
    complete_aq_data=a.complete_aq_data
    complete_aq_data_test=a.complete_aq_data_test
    
    for now_x,now_delta,now_y,now_time in a.next_train(48,16):
        print(now_time[0])
    """
    
