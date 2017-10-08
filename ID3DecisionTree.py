# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:12:16 2017

@author: visakh
"""

import sys
import math

def entropy(prob):
    entropy_value = -1*prob*math.log2(prob)
    return entropy_value

def probability_calc(data_set_for_prob):
    positive_count=0
    count=0
    for lines in data_set_for_prob:
        count+=1
        if lines[-1]=='positive':
            positive_count+=1 
    probab = (positive_count/count)
    return probab

def info_gain(data_set, feature,split_value):
    initial_entropy = entropy(probability_calc(data_set)) 
    split_data = splitter(data_set, feature,split_value)
    infogain = initial_entropy
    for j in range(0,len(split_data)):
        if len(split_data[j])>0:
            count_pos,count_neg = counter(split_data[j])
            if(count_pos == 0 or count_neg == 0):
                infogain = 0
            else:
                current_entropy=entropy(probability_calc(split_data[j]))
                prob2 = len(split_data)/len(data_set)
                infogain -= (prob2 * current_entropy)            
    return infogain            
    
def splitter(training_set_for_splitting,feature_for_split,split_point):
    split_data = {}
    if split_point == None:
        for features in feature_details[feature_for_split]:
            index_of_feature = feature_list.index(feature_for_split)
            i=0
            for feature_category in features:
                split_data[i] = []                
                for lines in training_set_for_splitting:
                    if (lines[index_of_feature]==feature_category):
                        split_data[i].append(lines)
                i=i+1
    else:
        for features in feature_details[feature_for_split]:
            index_of_feature = feature_list.index(feature_for_split)
            split_data[0]=[]
            split_data[1]=[]
            for lines in training_set_for_splitting:
                if(float(lines[index_of_feature])<=split_point):
                    split_data[0].append(lines)
                else:
                    split_data[1].append(lines)
    return split_data

def counter(training_set_for_counting):
    positive_no=0
    total_no=0
    for lines in training_set_for_counting:
        total_no+=1
        if lines[-1]=='positive':
            positive_no+=1     
    negative_no = total_no - positive_no
    return positive_no, negative_no

def subtree(training_set,recursion_level = 0):
    global tree
    global nominal_feature_flag
    tree = tree + "\n"
    for i in range(0,recursion_level):
        tree = tree + "\t"
    tree = tree + "|\t"
    flag = 0
    count_pos,count_neg = counter(training_set)
    if(not training_set):
        flag = 1
    if(count_pos == 0 or count_neg == 0):
        flag = 1        
    if ((count_pos + count_neg)<m):
        flag = 1
    if(flag == 1):
        randomnumber = 1
        #tree = tree + "["+ str(count_pos) +","+str(count_neg) + "]"
        #return count_pos,count_neg,"leaf"
    else:
        best_infogain = 0
        current_infogain=0
        best_feature = None
        for features in feature_list:
            index_of_feature = feature_list.index(features)
            if feature_details[features] == "real":
                values = []
                for lines in training_set:
                    values.append(lines[index_of_feature])
                threshhold = (float(max(values)) + float(min(values)))/2
                if(threshhold != float(max(values))):
                    current_infogain = info_gain(training_set,features,threshhold)
                else:
                    current_infogain = 0
            else:
                if(nominal_feature_flag[features] != 1):
                    current_infogain = info_gain(training_set,features,None)
                else:
                    current_infogain = 0
            if(current_infogain > best_infogain):
                best_infogain = current_infogain 
                best_feature = features
        if best_infogain <= 0:
            count_pos,count_neg = counter(training_set)
            #tree = tree + "["+ str(count_pos)+","+str(count_neg)+ "]"
            #return count_pos,count_neg,"leaf"
        else:
            flagging = ""
            index_of_best_feature = feature_list.index(best_feature)
            if feature_details[best_feature] == "real":
                values = []
                for lines in training_set:
                    values.append(lines[index_of_best_feature])                    
                best_threshhold = (float(max(values)) + float(min(values)))/2
                divided_tree = splitter(training_set,best_feature,best_threshhold)
            else:
                flagging = "nominal"
                nominal_feature_flag[best_feature]=1
                divided_tree = splitter(training_set,best_feature,None)
            recursion_level += 1
            for i in range(0,len(divided_tree)):
                if(flagging=="nominal"):
                    tree = tree + str(best_feature) + "=" + str(feature_details[best_feature][0][i])
                else:
                    if(i==0):
                        tree = tree + str(best_feature) + "<=" + str(best_threshhold)
                    else:
                        tree = tree + str(best_feature) + ">" + str(best_threshhold)
                c1,c2 = counter(divided_tree[i])
                tree = tree + " ["+ str(c1) + ","+ str(c2) +"]"
                subtree(divided_tree[i],recursion_level)
    
            

def readtrain(train_name):
    trainarff = open(train_name,'r')
    trainlines = []
    trainlines=trainarff.readlines()
    data_feature = []
    data_feature_value = {} 
    data_training = []        
    for line in trainlines:
        line = line.replace("}","")
        line = line.replace("\'","")
        if line.split()[0] == "@attribute":
            line = line.replace(",","")
            if not line.split()[1] == "class":
                data_feature.append(line.split()[1])
            data_feature_value[line.split()[1]] = []
            if line.split()[-1] == "real":                
                data_feature_value[line.split()[1]] = line.split()[-1]
            elif line.split()[2] == "{":
                data_feature_value[line.split()[1]].append(line.split()[3:])
        elif not line.startswith("@"):
            line=line.replace('\n',"")
            data_training.append(line.split(','))
    returned_training_data = (data_feature, data_feature_value,data_training)
    return returned_training_data


if __name__ == "__main__":
    #train_name = str(sys.argv[1])
    #test_name = str(sys.argv[2])
    #m = int(sys.argv[3])
    nominal_feature_flag = {}
    train_data = readtrain("heart_train.arff")
    feature_list = train_data[0]
    for features in feature_list:
        nominal_feature_flag[features]=0
    feature_details = train_data[1]
    initial_data = train_data[2]
    m=10
    tree = ""
    #k = splitter(initial_data,'exang',None)
    subtree(initial_data)
    print(tree)