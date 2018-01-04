# HelloTF

##Installing Tensorflow
Depending on your platform, you need to install tensorflow library. The below link should help.
```
https://www.tensorflow.org/install/
```

For this basic code, you do not need GPU support. It will easily run on a CPU. 

Also you need Python3.x

##Running the code
python3 1_gate_learn.py

##OUTPUT:
###Code Output: 
``` 
number of training examples = 4 
X_train shape: (2, 4) 
Y_train shape: (1, 4) 
0 0.594839 1.0 [[ 0.48101175  0.36990312  0.31571779  0.44649839]] 
10 0.451503 1.0 [[ 0.30754173  0.29681206  0.15319541  0.40467405]] 
20 0.394287 1.0 [[ 0.22161396  0.24571806  0.09625828  0.38950431]] 
30 0.344328 1.0 [[ 0.16325271  0.20063613  0.08151171  0.41442931]] 
40 0.294998 1.0 [[ 0.10710264  0.16709907  0.05555663  0.44216263]] 
50 0.254292 1.0 [[ 0.07689214  0.1468157   0.04326564  0.48218927]] 
60 0.218617 1.0 [[ 0.06239825  0.11634921  0.04796128  0.52672237]] 
70 0.18769 1.0 [[ 0.05102341  0.11169618  0.04297816  0.5802145 ]] 
80 0.159034 1.0 [[ 0.03196755  0.07987423  0.03033573  0.60754716]] 
90 0.139308 1.0 [[ 0.02612194  0.07000994  0.02878116  0.65850174]] 
100 0.123399 1.0 [[ 0.02204676  0.06331776  0.02755842  0.69188172]]
```
###Explained Op: 
After every 10 iterations, it will provide you the epoch #, log loss, accuracy, and output at the last layer.
It is trained only on 4 data points. 
I have used logic gates inputs and output. 

###Suggested modifications
```
Ln20: Y_train = np.matrix('0 ;0 ;0 ;1')    #try changing the gate type
```
You can change the values here. The above example resembles AND gate. You can try other ones.

```
Ln42: h = 4   
```
This variable specifies the number of hidden layer nodes. Try changing it and see the effect.

```
Ln77: optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.3).minimize(cost)  
```
Try changing the learning rate here. Try changing it and see the effect.

