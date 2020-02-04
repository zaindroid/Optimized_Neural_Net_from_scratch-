import numpy
import scipy.special
import matplotlib.pyplot


class Neural_Net:
    def __init__(self,no_input_nodes,no_hidden_nodes,no_output_nodes,learning_rate):
         #initialize the neural newtwork
        self.i_nodes=no_input_nodes
        self.h_nodes=no_hidden_nodes
        self.o_nodes= no_output_nodes
        self.l_rate=learning_rate
           # link weight matrices, wih and who
           # weights inside the arrays are w_i_j, where link i from node i to node j in the next layer 
           # w11 w21 
           # w12 w22 etc

    
        self.w_ih=numpy.random.normal(0.0,pow(self.h_nodes,-0.5),(self.h_nodes, self.i_nodes)) 
        self.w_ho=numpy.random.normal(0.0,pow(self.o_nodes,-0.5),(self.o_nodes, self.h_nodes)) 
            
        # activation function is the sigmoid function
       # self.activation_function = lambda x:scipy.special.expit(x
        self.acti_func= lambda x: scipy.special.expit(x)
    
    
        pass                                        
    def train(self,in_list,labels):
        #train the neural network
    
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
    #Query the neural network for output
        inputs= numpy.array(in_list,ndmin=2).T
         # signals in first layer          
        hidden_in=numpy.dot(self.w_ih,inputs)
        hidden_out=self.acti_func(hidden_in)
        # signals in final layer
        final_in=numpy.dot(self.w_ho,hidden_out)
        final_out=self.acti_func(final_in)
        
        return final_out

    
 ########################################3
#in_nodes=784
#out_nodes=10
#hidden_nodes=150
#lr=0.4
#nn=Neural_Net(in_nodes,hidden_nodes,out_nodes,lr) 
    
    
    
    ####################Training############
    
# load the mnist training data CSV file into a list
training_data_file = open("mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# epochs is the number of times the training data set is used for training
epochs = 10

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(out_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        nn.train(inputs, targets)
        pass
    pass


######################## Testing ######################
    



test_data_file = open("mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = nn.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
#print ("performance = ", scorecard_array.sum() / scorecard_array.size)
#
#print(nn.w_ih.size)
#print(nn.w_ho.size)

    