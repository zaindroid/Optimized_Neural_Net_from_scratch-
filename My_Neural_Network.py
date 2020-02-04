# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 09:33:29 2019

@author: Zain Ul Haq
"""

''' ============================ SECTION-1 (Data extraction)======================================================='''


import os
import numpy as np
import numpy
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import platform


def rgb2gray(rgb):
    '''Function for converting rgb to greyscale images'''
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def load_pickle(f):
    '''function for loading picke data as dictornary'''
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ function for loading single batch of cifar-10 dataset """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3072)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ functon for loading all cifar-10 data and combine as one """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    '''Function for spllitting Cifar-10 data into training, validating and testing example sets '''
    cifar10_dir = 'cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    
    # Subsample the data
    mask = range(num_training, num_training + num_validation)  # setting range for data splitting
    X_val = X_train[mask]                                       #splitting to validation example data
    y_val = y_train[mask]                                       #splitting to validation lables data
    mask = range(num_training)
    X_train = X_train[mask]                               #splitting to training example data
    y_train = y_train[mask]                               #splitting to training lables data
    mask = range(num_test)
    X_test = X_test[mask]                                 #splitting to testing example data 
    y_test = y_test[mask]                                 #splitting to testing lables data
    x_train = X_train.astype('float32')                # converting the data to float type array
    x_test = X_test.astype('float32')                  # converting the data to float type array
    x_val= X_val.astype('float32')                     # converting the data to float type array
    x_train /= 255                                    #Normalizing the data between range of 0-1
    x_test /= 255                                     #Normalizing the data between range of 0-1
    #x_train=np.arange(num_training)
  

    return x_train, y_train, x_val, y_val, x_test, y_test

 
# Invoking the above function to get our data.
x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()    

# Prininting all the data after splitting
print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)



''' ============================ SECTION-2 (Neural Network acrchitecture)======================================================='''


class Neural_Net:
    def __init__(self,no_input_nodes,no_hidden_nodes,no_output_nodes,learning_rate):
         #initialize the neural newtwork
        self.i_nodes=no_input_nodes
        self.h_nodes=no_hidden_nodes
        self.o_nodes= no_output_nodes
        self.l_rate=learning_rate
         

            #Random initialization of the layers weights 
        self.w_ih=numpy.random.normal(0.0,pow(self.h_nodes,-0.5),(self.h_nodes, self.i_nodes)) 
        self.w_ho=numpy.random.normal(0.0,pow(self.o_nodes,-0.5),(self.o_nodes, self.h_nodes)) 
            
        # activation function is the sigmoid function
       
        self.acti_func= lambda x: scipy.special.expit(x)    # Returns sigmoid of the input
    
    
        pass                                        
    def train(self,in_list,labels):
        '''#train the neural network'''
    
        inputs= numpy.array(in_list,ndmin=2).T
        target= numpy.array(labels,ndmin=2).T
        
         # signals in first layer          
        hidden_in=numpy.dot(self.w_ih,inputs)
        hidden_out=self.acti_func(hidden_in)
        # signals in final layer
        final_in=numpy.dot(self.w_ho,hidden_out)
        final_out=self.acti_func(final_in)
        out_err=target-final_out
        hidden_err=numpy.dot(self.w_ho.T,out_err)
        
        #updating weights
        self.w_ho+=self.l_rate*numpy.dot((out_err*final_out*(1-final_out)),numpy.transpose(hidden_out))
        self.w_ih+=self.l_rate*numpy.dot((hidden_err*hidden_out*(1-hidden_out)),numpy.transpose(inputs))
        pass
    def query(self,in_list):
        '''calculates feedforward for the neural network and produces output'''
        inputs= numpy.array(in_list,ndmin=2).T
         # signals in first layer          
        hidden_in=numpy.dot(self.w_ih,inputs)
        hidden_out=self.acti_func(hidden_in)
        # signals in final layer
        final_in=numpy.dot(self.w_ho,hidden_out)
        final_out=self.acti_func(final_in)
        
        return final_out
    

''' ============================ SECTION-3 (Data preprocessing and Model training)======================================================='''
## initiating the nueral network instance for our data 
in_nodes=1024   # No. of nodes in input layer equal to 1D array of greyscaled image
out_nodes=10     # No. of nodes in output layer equal to 10 classes
hidden_nodes=2000  # No. of nodes in hidden layer chosen based on hit and trial basis
lr=0.28              # learning rate for converging 
nn=Neural_Net(in_nodes,hidden_nodes,out_nodes,lr)   ## instantiate the object of class neural_network for cifar-10 model fitting


## training the neural network
epochs=5000    # no of looping through examples sets
num=0         # for tracking the number of training example in dataset
for e in range(epochs):
   for items in y_train:
       if num==len(y_train):      #check for index overflow range
            break
       targets = np.zeros(out_nodes) + 0.01     #creating target array for training with no.of classes all indices initialized to 0.01
       targets[int(items)] = 0.99               # declaring traget class index to 0.99 as the correct label for training
       single_img=np.asfarray(x_train[num])     # exatrcting single row of training example as input to training function
       single_img=np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0)) # converting image to 32x32x3 format RGB image
       single_img=rgb2gray(single_img)              # convertng image to greyscale
       single_img=single_img.reshape(-1)            # converting greyscale to 1D array
       inputs = (single_img *  0.99) + 0.01          # scaling array in range 0.01 to 1 for neural network training avoiding 0 weights update 
       nn.train(single_img, targets)                # trainig the neural network
       num+=1 



''' ============================ SECTION-4 (Model Testing)======================================================='''
#### =======================Model testing on validation ==================================================== 
scorecard_val = []
num=0
# go through all the records in the test data set
for record in y_val:
    # split the record by the ',' commas
    if num==len(y_val):
      break
    # correct answer is first value
    correct_label = int(record)
    # scale and shift the inputs
   
    single_img=np.asfarray(x_val[num])
    single_img=np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
    single_img=rgb2gray(single_img)
    single_img=single_img.reshape(-1)
    num+=1
    # query the network
    outputs = nn.query(single_img)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard_val.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard_val.append(0)
        pass
    
    pass


#### ======================= Model testing on test examples ==================================================== 
    
scorecard_test = []
num=0
# go through all the records in the test data set
for record in y_test:
    # split the record by the ',' commas
    if num==len(y_test):
      break
    # correct answer is first value
    correct_label = int(record)
    # scale and shift the inputs
   
    single_img=np.asfarray(x_test[num])
    single_img=np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
    single_img=rgb2gray(single_img)
    single_img=single_img.reshape(-1)
    num+=1
    # query the network
    outputs = nn.query(single_img)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard_test.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard_test.append(0)
        pass
    
    pass

#### ======================= Model testing on train examples ==================================================== 
    
scorecard_train = []
num=0
# go through all the records in the test data set
for record in y_train:
    # split the record by the ',' commas
    if num==len(y_train):
      break
    # correct answer is first value
    correct_label = int(record)
    # scale and shift the inputs
   
    single_img=np.asfarray(x_train[num])
    single_img=np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
    single_img=rgb2gray(single_img)
    single_img=single_img.reshape(-1)
    num+=1
    # query the network
    outputs = nn.query(single_img)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard_train.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard_train.append(0)
        pass
    
    pass



''' ============================ SECTION-5 (Model Peformance)======================================================='''

scorecard_array_val = numpy.asarray(scorecard_val)
print ("performance on validation set= {:f} % ".format(scorecard_array_val.sum() / scorecard_array_val.size*100))


scorecard_array_test = numpy.asarray(scorecard_test)
print ("performance on test set = {:f} % ".format(scorecard_array_test.sum() / scorecard_array_test.size*100))

scorecard_array_train = numpy.asarray(scorecard_train)
print ("performance on training set= {:f} % ".format(scorecard_array_train.sum() / scorecard_array_train.size*100))



