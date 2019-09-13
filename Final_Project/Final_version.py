
# coding: utf-8

# load library
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

from sklearn.datasets import load_svmlight_file # for reading dataset
from sklearn.model_selection import train_test_split # for train and test dataset
from sklearn.feature_extraction.text import TfidfTransformer # for calculating tf-idf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import coo_matrix
# from scipy.sparse import csr_matrix # convert numpy matrix to csr_matrix
# from sklearn.svm import SVC # for SVM

from functools import reduce
from collections import defaultdict
from tqdm import tqdm # for observing iterations and runing time

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# load data
x_train, y_train = load_svmlight_file("rcv1subset_topics_train_1.svm", multilabel=True)
x_test, y_test = load_svmlight_file("rcv1subset_topics_test_1.svm", multilabel=True)

# data clean
x_test = x_test[:, 0:x_train.shape[1]]
sp.sparse.save_npz('rcv1_1_test', x_test) # save to .npz data
# x_train = sp.sparse.load_npz('yease_train_data.npz')

# convert y to CSR Matrix
def sparse_y(y_train):
    l = [(i,j) for i,j in enumerate(y_train)]
    row, column = list(zip(*((x, int(z)) for x,y in l for z in y)))
    data = [1]*len(row)
    return coo_matrix((data, (row, column)), (max(row)+1,max(column)+1), dtype=np.int8)

y_new = sparse_y(y_train)


# process list with duplicated number for different values
## process count list with duplicated numbers; return: [(number set), [[index1..], [index2..], ..]
def duplicate_list(count):
    ls = defaultdict(list)
    for i, j in enumerate(count):
        ls[j].append(i)
    return list(zip(*[(number, index) for number,index in ls.items()]))

## process for list with tie in top k largest values
def deduplicate_max(index, index0, ls, n):
    r_old = ls
    r = []
    if len(ls) >= n:
        return list(np.random.choice(ls, n, replace=False))
    else:
        while len(r) < n:
            r += r_old
            index0.remove(max(index0))
            if len(index0) > 0:
                r_new = index[1][index[0].index(max(index0))]
                if len(r)+len(r_new) > n:
                    return r + list(np.random.choice(r_new, n-len(r), replace=False))
                    break
            else:
                return r
            r_old = r_new

# calculate a,b,c,d for Chi-Test
def feature_label(feature, label, n):
    a = 0
    b = 0
    c = 0
    d = 0
    index1 = list(x_train[:,feature].nonzero()[0])
    index2 = [x for x in range(n) if x not in index1]
    
    for y in index1:
        if label in y_train[y]:
            a += 1
        else:
            b += 1
    
    for y in index2:
        if label in y_train[y]:
            c += 1
        else:
            d += 1
    return a, b, c, d

# Chi-Square Test
def Chi_Test(x_train, label_number):
    n = x_train.shape[0]
    chi_test = []
    for i in tqdm(range(x_train.shape[1])):
        chi = []
        for j in range(label_number):
            a,b,c,d = feature_label(i, j, n)
            if (a+b)*(a+c)*(b+d)*(c+d) > 0:
                chi.append(n*(a*d-b*c)/((a+b)*(a+c)*(b+d)*(c+d)))
            else:
                chi.append(0)
        chi_test.append(max(chi))
    return chi_test


# feature selection
def Feature_Selection(x_train, f, tf=False, chi=False):
    if chi == True:
        x_new = SelectKBest(chi2, k=f)
        x_new.fit_transform(x_train, y_new)
        s = list(x_new.get_support(indices=True))
        # s = Chi_Test(x_train) # chi=True: Chi-Test
    else:
        
        if tf == True:
            tfidf_colmax = x_train.max(axis=0) # tf=True: data has already been applied with tf-idf
            s = list(tfidf_colmax.data)
        else:
            ti = TfidfTransformer() # build object (l-norm=2, smooth_idf=True)
            tfidf = ti.fit_transform(x_train) # calculate tf-idf for train data
            tfidf_colmax = tfidf.max(axis=0) # calculate max value for each feature (csr)
            s = list(tfidf_colmax.data)
    
    #  preprocessing for selecting top f largest values
    s_index = duplicate_list(s)
    s_index0 = list(s_index[0])
    feature = s_index[1][s_index[0].index(max(s_index0))]
        
    return deduplicate_max(s_index, s_index0, feature, f)

# subset S out of training data
def Construct_S(x_train, feature):
    return list(set(i for i,j in zip(*x_train.nonzero()) if j in feature))

# similarity 1: Jaccard Similarity
def Similarity_J(t, x, nt):
    nx = x.nnz
    n_common = len(set(t.nonzero()[1])&set(x.nonzero()[1]))
    return 1-(nt+nx-2*n_common)/(nt+nx-n_common) # S1&S2)/(S1|S2)

# similarity 2: Cosine Similarity
def Similarity_C(t, x):
    return cosine_similarity(t, x).item() # <t,x>/(2-norm(t)*2-norm(x))

# KNN for each testing document (t)
def KNN(t, S, k, cos=False):
    N = len(S) # size of S
    nt = t.nnz # number of features in testing document
    s = []
    for i in range(N):
        if cos == True:
            s.append(Similarity_C(t, x_train[S[i]]))
        else:
            s.append(Similarity_J(t, x_train[S[i]], nt))
        
    # preprocessing for calculation of top k largest values
    s_index = duplicate_list(s)
    s_index0 = list(s_index[0])
    neighbor = s_index[1][s_index[0].index(max(s_index0))]
    
    return deduplicate_max(s_index, s_index0, neighbor, k)

# predict test labels using KNN-top m labels
def Label_M(knn, y_train, m):
    label_each = list(map(lambda i: y_train[i], knn)) # labels for each knn
    label_all = reduce(lambda p,q: p+q, label_each) # all labels for knn
    label_set = set(label_all) # nonduplicate set of labels for knn
    count = [] # temp for frequence
    label = [] # temp for label
    label_test = [] # predicted labels
    
    for l in label_set:
        label.append(l)
        count.append(label_all.count(l)) # frequence corresponding to label
    
    # case 1: all count numbers are 1
    if max(count) == 1:
        return [np.random.choice(label, replace=False)]
    else:
        count_index = duplicate_list(count)
        frequent1_index = count_index[1][count_index[0].index(max(count_index[0]))]
        for j in frequent1_index:
            label_test.append(label[j]) # case 2: store labels with the most frequences first
        if len(label_test) < m:
            try:
                frequent2_index = count_index[1][count_index[0].index(max(count_index[0])-1)]
                for k in frequent2_index:
                    label_test.append(label[k]) # case 3: store labels with the submost frequences next
            except ValueError:
                return label_test
        return label_test

# predict test labels using score model
def Label_S(t, knn, label_number, bar=0.03):
    score = []
    for i in range(label_number):
        temp = []
        for j in knn:
            if i in y_train[j]:
                temp.append(cosine_similarity(t, x_train[j]).item())
            else:
                temp.append(0)
        score.append(sum(temp))
    count = [1 if i >= bar else 0 for i in score]
    count_index = duplicate_list(count)
    return count_index[1][count_index[0].index(max(count_index[0]))]

# measure with expected average accuracy
def Measure(y_predict, y_test, bar=0.5):
    size = len(y_test)
    flag = 0
    for i in range(size):
        count = 0
        for j in range(len(y_predict[i])):
            if y_predict[i][j] in y_test[i]:
                count += 1
        if count/len(y_predict[i]) >= bar:
            flag += 1
    return flag/size


# run for experiment
def run(k, f, m=3, tf=True, chi=True, cos=False):
    start_time = time.clock()
    feature = Feature_Selection(x_train, f=f, tf=True, chi=chi)
    S = Construct_S(x_train, feature)
    test_number = x_test.shape[0]
    test1 = []
    test2 = []
    for i in range(test_number):
        knn = KNN(x_test[i], S, k=k, cos=cos)
        test1.append(Label_M(knn, y_train, m))
        test2.append(Label_S(x_test[i], knn, label_number))
    end_time = time.clock()
    runtime = end_time - start_time
    return Measure(test1, y_test), Measure(test2, y_test), runtime

    