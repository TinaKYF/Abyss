# coding: utf-8

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2.cv2 as cv2
from sklearn.naive_bayes import BernoulliNB
import random
import sys

# In[4]:


#import good and flare image samples
def import_data(path, file_format):
    data_list = []
    for root, dirs, files in os.walk(path):
        for a_file in files:
            if a_file.endswith("."+file_format):
                data_list.append(os.path.join(root, a_file))
    return data_list


# In[6]:


#convert images into data for later processing
def convert_data(data_list):
    data_lena_list = []
    for data in data_list:
        data_pro = cv2.imread(data)
        data_lena_list.append(data_pro)
    return data_lena_list


# In[7]:


#convert image data into histogram, and use only two color channels 0 and 1 which corresponds to "B" and "G"
def process_data(data_list):
    hist_data_list = []
    for data in data_list:
        image = cv2.resize(data,(256,256),interpolation=cv2.INTER_CUBIC)
        hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0,256.0,0.0,256.0])
        hist_data_list.append(((hist/256).flatten()))
    return hist_data_list


# In[9]:


#get Y values for both good data and flare data, for which good data uses 0 for representation and flare data uses 1
def set_label(data_list, label):
    Y = []
    for data in data_list:
        Y.append(label)
    return Y


# In[10]:


#divide the whole dataset X into 5 parts
def separate_dataset(x_set, num_parts):
    result = []
    ele_each_set = len(x_set)/num_parts
    count = 0
    for i in range(0, num_parts):
        list_1 = []
        while count < ((i+1)*ele_each_set):
            list_1.append(x_set[count])
            count = count+1
        result.append(list_1)
    return result


# In[11]:


#separate X and Y
def separate_X_Y(total_list):
    X_total = []
    Y_total = []
    for ele in total_list:
        X_total.append(ele[0])
        Y_total.append(ele[1])
    return (X_total, Y_total)


# In[12]:


#use the nth element as the test and the rest as training
def find_test_train(total_list, nth_ele):
    test_list = total_list[nth_ele]
    X_test, Y_test = separate_X_Y(test_list)
    total_list_copy = list(total_list)
    total_list_copy.pop(nth_ele)
    X_train = []
    Y_train = []
    for total_sep in total_list_copy:
        X_train_ele, Y_train_ele = separate_X_Y(total_sep)
        X_train = X_train + X_train_ele
        Y_train = Y_train + Y_train_ele
    return (X_test, Y_test, X_train, Y_train)



# In[21]:


def detectory(image):
    if not os.path.exists(image):
        return "the image does not exist, please check the path"
    else:
        good_data_list = import_data("./data/good-data", "JPG")
        flare_data_list = import_data("./data/flare-data", "JPG")
        good_data_lena_list = convert_data(good_data_list)
        flare_data_lena_list = convert_data(flare_data_list)
        hist_good_data_list = process_data(good_data_lena_list)
        hist_flare_data_list = process_data(flare_data_lena_list)
        #set good data label to 0 and set flare data label to 1
        Y_good_data_list = set_label(hist_good_data_list, "0")
        Y_flare_data_list = set_label(hist_flare_data_list, "1")
        #convert both good data list and flare data list to np.array for later processing
        X_good = np.array(hist_good_data_list)
        X_flare = np.array(hist_flare_data_list)
        #convert y_hist_good_list to array Y_good
        Y_good = np.array(Y_good_data_list)
        #convert y_hist_flare_list to array Y_flare
        Y_flare = np.array(Y_flare_data_list)
        #combine X_good and X_flare together as the whole set, and combine Y_good and Y_flare together as the whole set
        X = np.concatenate((X_good, X_flare))
        Y = np.concatenate((Y_good, Y_flare))
        zipped_list = zip(X.tolist(), Y.tolist())
        random.shuffle(zipped_list)
        X_train, Y_train = separate_X_Y(zipped_list)
        naive_bayes = BernoulliNB()
        result = naive_bayes.fit(X_train, Y_train)
        image_data_list = convert_data([image])
        image_data_lena_list = process_data(image_data_list)
        predict = result.predict(np.array(image_data_lena_list))
        return predict[0]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print 'Usage: python path_of_the_image/image.JPG'
        exit(1)
    output = detectory(sys.argv[1])
    print output
