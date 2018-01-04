import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   #to supress warnings


tf.set_random_seed(20)                  
np.random.seed(20)


# Loading the dataset
X_train = np.matrix('0 0; 0 1;1 0;1 1')
X_train = X_train.transpose()        #we store records column wise


Y_train = np.matrix('0 ;0 ;0 ;1')    #try changing the gate type
Y_train = Y_train.transpose()        #we store records column wise


print ("number of training examples = " + str(X_train.shape[1]))

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))


n_x = 2 #input features
n_y = 1 #output classes


ops.reset_default_graph()   # to be able to rerun the model without overwriting tf variables

#create_placeholders
X = tf.placeholder(tf.float32,shape=(n_x,None),name="X")
Y = tf.placeholder(tf.float32,shape=(n_y,None),name="Y")


#initialize hidden layers 
h = 4   #nodes in hidden layer   #try changing number of hidden layer nodes
W1 = tf.get_variable("W1", [h,2], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b1 = tf.get_variable("b1", [h,1], initializer = tf.zeros_initializer())
W2 = tf.get_variable("W2", [1,h], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
b2 = tf.get_variable("b2", [1,1], initializer = tf.zeros_initializer())

#Dictionary makes it easy to pass variables into functions
parameters = {"W1": W1,"b1": b1, "W2": W2,"b2": b2}


#Define the structure of the neural network
def forward_propagation(X, parameters):
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	Z1 = tf.add((tf.matmul(W1,X)),b1)
	A1 = tf.nn.relu(Z1) 
	Z2 = tf.add((tf.matmul(W2,A1)),b2)   
	return Z2

#function to calculate cost 
def compute_cost(Z2, Y):
	logits = tf.transpose(Z2)
	labels = tf.transpose(Y)
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
	return cost

init = tf.global_variables_initializer()

Z2 = forward_propagation(X, parameters)
cost  = compute_cost(Z2, Y)
correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.3).minimize(cost)       #try changing learning rate

#to get output for all input records
accuracy_evaluation = tf.sigmoid(tf.add((tf.matmul(W2,tf.nn.relu(tf.add((tf.matmul(W1,X)),b1)))),b2))   

#structuring complete, now execution time
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(100+1):
		_ , bcost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
		if epoch % 10 ==0:
			#print after every specific interations
			print(epoch, bcost,accuracy.eval({X: X_train, Y: Y_train}),sess.run(accuracy_evaluation, feed_dict = {X: X_train}))