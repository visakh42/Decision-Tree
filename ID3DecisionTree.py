# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:12:16 2017

@author: visakh
"""

import sys
import math

def entropy(prob):
    entropy_value = -1*((prob*math.log2(prob)) + ((1-prob)*math.log2(1-prob)))
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
            if not (count_pos == 0 or count_neg == 0):
                current_entropy=entropy(probability_calc(split_data[j]))
                prob2 = len(split_data[j])/len(data_set)
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

def candidate_split(instance_set,feature):
    values = []
    candidates = []
    unique_values = []
    index_of_feature = feature_list.index(feature)
    for lines in instance_set:
        values.append(lines[index_of_feature])
    unique_values = list(set(values))
    unique_values = sorted(unique_values)
    for i in range(0,len(unique_values)-1):
        candidates.append((float(unique_values[i])+float(unique_values[i+1]))/2)
    return candidates
        
def counter(training_set_for_counting):
    positive_no=0
    total_no=0
    for lines in training_set_for_counting:
        total_no+=1
        if lines[-1]=='positive':
            positive_no+=1     
    negative_no = total_no - positive_no
    return positive_no, negative_no

def subtree(training_set,recursion_level = 0,decisiontree='start'):
    global nodecollection
    global tree
    global nominal_feature_flag
    global best_threshhold
    flag = 0
    count_pos,count_neg = counter(training_set)
    if(not training_set):
        flag = 1
    if(count_pos == 0 or count_neg == 0):
        flag = 1 
    if ((count_pos + count_neg)<m):
        flag = 1
    if(flag == 1):
        if(count_pos>count_neg):
            tree = tree + "    Positive"
            node= [decisiontree,"Positive",0,0]
            nodecollection.append(node)
        else:
            tree = tree + "    Negative"
            node = [decisiontree,"Negative",0,0]
            nodecollection.append(node)
    else:
        best_infogain = 0
        current_infogain=0
        best_feature = None
        threshhold = {}
        for features in feature_list:
            candidate_infogain=0
            threshhold[features] = None
            if feature_details[features] == "real":
                candidate_splits = candidate_split(training_set, features)
                for candidates in candidate_splits:
                    candidate_infogain = info_gain(training_set,features,candidates)
                    if((candidate_infogain > current_infogain) & (candidates not in best_threshhold[features])):
                        current_infogain = candidate_infogain
                        threshhold[features]=candidates
            else:
                if(nominal_feature_flag[features] != 1):
                    current_infogain = info_gain(training_set,features,None)
                else:
                    current_infogain = 0
            if(current_infogain > best_infogain):
                best_infogain = current_infogain 
                best_feature = features
                best_threshhold[features].append(threshhold[features])
        if best_infogain <= 0:
            count_pos,count_neg = counter(training_set)
            if(count_pos>count_neg):
                tree = tree + "    Positive"
                node=[decisiontree,"Positive",0,0]
                nodecollection.append(node)
            else:
                tree = tree + "    Negative"
                node=[decisiontree,"Negative",0,0]
                nodecollection.append(node)
        else:
            flagging = ""
            recursion_level += 1
            if feature_details[best_feature] == "real":
                divided_tree = splitter(training_set,best_feature,threshhold[best_feature])
            else:
                flagging = "nominal"
                #nominal_feature_flag[best_feature]=1
                divided_tree = splitter(training_set,best_feature,None)
            for i in range(0,len(divided_tree)):
                tree = tree + "\n"
                for j in range(0,recursion_level):
                    tree = tree + "|  "
                if(flagging=="nominal"):
                    factor = "="
                    split_condition = feature_details[best_feature][0][i]
                    tree = tree + str(best_feature) + "=" + str(feature_details[best_feature][0][i])
                else:
                    if(i==0):
                        factor = "<="
                        split_condition = threshhold[best_feature]
                        tree = tree + str(best_feature) + "<=" + str(threshhold[best_feature])
                    else:
                        factor = ">"
                        split_condition = threshhold[best_feature]
                        tree = tree + str(best_feature) + ">" + str(threshhold[best_feature])
                c1,c2 = counter(divided_tree[i])
                tree = tree + " ["+ str(c1) + ","+ str(c2) +"]"
                node=[decisiontree,best_feature,factor,split_condition]
                nodecollection.append(node)
                subtree(divided_tree[i],recursion_level,node)     
                
def classification(data_line,parentnode="start"):
    for nodes in nodecollection:
        if nodes[0] == parentnode:
            if nodes[1]=="Positive":
                print(data_line,": Positive")
            elif nodes[1]=="Negative":
                print(data_line,": Negative")
            else:
                indexer= feature_list.index(nodes[1])
                if nodes[2]== "=":
                    if nodes[3] == data_line[0][indexer]:
                        classification(data_line,nodes)
                elif nodes[2]==">":
                    if float(nodes[3]) < float(data_line[0][indexer]):
                        classification(data_line,nodes)
                elif nodes[2]=="Positive":
                    print(data_line,": Positive")
                elif nodes[2]=="Negative":
                    print(data_line,": Negative")
                elif nodes[2]== "<=": 
                    if float(nodes[3]) >= float(data_line[0][indexer]):
                        classification(data_line,nodes)
            
                    
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
    nodecollection = []
    nominal_feature_flag = {}
    best_threshhold = {}
    train_data = readtrain("heart_train.arff")
    feature_list = train_data[0]
    for features in feature_list:
        nominal_feature_flag[features]=0
        best_threshhold[features] = []
    feature_details = train_data[1]   
    initial_data = train_data[2]
    m=10
    tree = ""
    subtree(initial_data)
    classification([initial_data[5]])
    print(tree)

    
