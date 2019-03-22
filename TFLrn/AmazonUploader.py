# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:00:18 2019

@author: DevAccessa
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
def rhead(x, nrow = 6, ncol = 4):
    pd.set_option('display.expand_frame_repr', False)
    seq = np.arange(0, len(x.columns), ncol)
    for i in seq:
        print(x.loc[range(0, nrow), x.columns[range(i, min(i+ncol, len(x.columns)))]])
    pd.set_option('display.expand_frame_repr', True)

housing_bunch = fetch_california_housing()
housing_target = np.c_[housing_bunch.data,housing_bunch.target]
column_names = np.concatenate((housing_bunch.feature_names,["target"]))
housing = pd.DataFrame(housing_target,columns=column_names)

# since we are manually doing the normal equation, we do like so
m,n = housing_bunch.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing_bunch.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)),scaled_housing_data]
#X = tf.constant(housing_data_plus_bias,dtype=tf.float32,name="X")
#y = tf.constant(housing_bunch.target.reshape(-1,1),dtype=tf.float32,name="y")
#XT = tf.transpose(X)
#theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)


def localized_code():
    X = tf.constant(scaled_housing_data_plus_bias,dtype=tf.float32,name="X")
    y = tf.constant(housing_bunch.target.reshape(-1,1),dtype=tf.float32,name="y")
    
    theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0), name="theta")
    
    y_pred = tf.matmul(X,theta,name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    #gradients = 2/m*tf.matmul(tf.transpose(X),error)
    gradients = tf.gradients(mse,[theta])[0]
    learning_rate = 0.01
    n_epochs = 1000
    training_op = tf.assign(theta,theta-learning_rate*gradients)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE=",mse.eval())
            sess.run(training_op)
            
        best_theta = theta.eval()
        
    
    
    
    reset_graph()
    
    a = tf.Variable(0.2, name="a")
    b = tf.Variable(0.3, name="b")
    z = tf.constant(0.0, name="z0")
    for i in range(100):
        z = a * tf.cos(z + i) + z * tf.sin(b - i)
    
    grads = tf.gradients(z, [a,b])
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        init.run()
        print(z.eval())
        print(sess.run(grads))
        
    # using tensorboard
    reset_graph()



from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir,now)

n_epochs = 10
batch_size = 100
learning_rate = 0.01

n_batches = int(np.ceil(m/batch_size))

X = tf.placeholder(tf.float32, shape=(None,n+1),name="X")
y = tf.placeholder(tf.float32, shape=(None,1),name="y")
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0,seed=42,name="theta"))
y_pred = tf.matmul(X,theta,name="predictions")
#create a name scope
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE',mse)
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())


def fetch_batch(epoch, batch_index, batch_size, n_batches, data, target):
    total_rows = np.size(target)
    np.random.seed(epoch*n_batches + batch_index )
    indices = np.random.randint(total_rows,size=batch_size)
    X_batch = data[indices]
    y_batch = target.reshape(-1,1)[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch,batch_index,
                                           batch_size,n_batches,
                                           scaled_housing_data_plus_bias,
                                           housing_bunch.target)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
    best_theta = theta.eval()

file_writer.close()


#reset_graph()
#init = tf.global_variables_initializer()
#def relu(X):
#    with tf.name_scope("relu"):
#        w_shape = (int(X.get_shape()[1]), 1)                          # not shown in the book
#        w = tf.Variable(tf.random_normal(w_shape), name="weights")    # not shown
#        b = tf.Variable(0.0, name="bias")                             # not shown
#        z = tf.add(tf.matmul(X, w), b, name="z")                      # not shown
#        return tf.maximum(z, 0., name="max")                          # not shown
#
#n_features = 3
#X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
#relus = [relu(X) for i in range(5)]
#output = tf.add_n(relus, name="output")
#file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
#
#X_sample = np.random.normal(size=(1,n_features))
#with tf.Session() as sess:
#    sess.run(init)
#    sess.run(output,feed_dict={X:X_sample})
#file_writer.close()

from sklearn.datasets import make_blobs
data = make_blobs(n_samples=50,n_features=2,centers=2,random_state=75)

reset_graph()

# simple linear regression example. One manual, another with estimator API
def simple_linear_regression_example():  
    sample_size = 1000000
    batch_size = 10
    n_epochs = 10
    x_data = np.linspace(0,10,sample_size)+np.random.uniform(-1.5,1.5,sample_size)
    y_label = np.linspace(0,10,sample_size)+np.random.uniform(-1.5,1.5,sample_size)
    
    xdf = pd.DataFrame(x_data,columns=['X'])
    ydf = pd.DataFrame(y_label,columns=['Y'])
    x_y = pd.concat((xdf,ydf),axis=1)
    
    m,c = np.random.rand(2)
    print('original slope:',m,' original intercept:',c)
    m = tf.Variable(m,dtype=tf.float32)
    c = tf.Variable(c,dtype=tf.float32)
    
    xph = tf.placeholder(tf.float32,[batch_size])
    yph = tf.placeholder(tf.float32,[batch_size])
    
    y_predicted = m*xph + c
    
    loss = tf.reduce_sum(tf.square(y_predicted-yph))
        
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for _ in range(n_epochs):
            rand_index = np.random.randint(len(x_data),size=batch_size)
            x_batch = x_data[rand_index]
            y_batch = y_label[rand_index]
            sess.run(train,feed_dict={xph:x_batch,yph:y_batch})
        final_slope,final_intercept = sess.run([m,c])    
        print('slope:',final_slope,' intercept:',final_intercept)
    x_y.sample(n=250).plot(kind='scatter',x='X',y='Y')
    y_final = [final_slope*x + final_intercept for x in x_data]
    plt.plot(x_data,y_final,'r')
    # end simple linear regression
    
    ### Estimator API : simple example
    feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
    estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)
    
    x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_label,
                                                        test_size=0.3, 
                                                        random_state=101)
    
    ## create some input functions
    input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,
                                                    batch_size=8,
                                                    num_epochs=None,
                                                    shuffle=True)
    train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},
                                                           y_train,
                                                           batch_size=8,
                                                           num_epochs=1000,
                                                           shuffle=False)
    eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},
                                                           y_eval,
                                                           batch_size=8,
                                                           num_epochs=1000,
                                                           shuffle=False)
    estimator.train(input_fn=input_func, steps=1000)
    train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)   
    eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)
    print('TRAINING DATA METRICS')
    print(train_metrics)
    print('EVAL METRICS')
    print(eval_metrics)   
    # lets predict new values
    brand_new_data = np.linspace(0,10,10)
    input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},
                                                          shuffle=False)
    predictions = []
    for pred in estimator.predict(input_fn=input_fn_predict):
        predictions.append(pred['predictions'])
    plt.figure()
    x_y.sample(n=250).plot(kind='scatter',x='X',y='Y')
    plt.plot(brand_new_data,predictions,'r')

### Classification example
# pima indians dataset
## predict this Class. Since the only possible values are 0,1 : this 
## the dataset has a Class column. this is our target. We are trying to 
## is a binary classification problem
diabetes = pd.read_csv('pima-indians-diabetes.csv')
## lets make a list of columns to normalize
# I dont want to normalize Class because that is my label column (the one I am trying to predict)
# I dont want to normalize Group because those are strings
# I dont want to normalize Age because I am going to convert it to a categorical column (as an example)
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 
                'Blood_pressure', 'Triceps',
                'Insulin', 'BMI', 'Pedigree']
# we could use sklearn MinMaxScaler to normalize but here we will show a 
# shortcut was to normalize multiple columns directly from pandas
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(
        lambda x:(x-x.min())/(x.max() - x.min())
        )
# now lets make our feature columns in preparation for estimator API
# for now, we will treat Age as a continuous(i.e. numeric) feature (we will convert it to categorical later)
# in case of a huge dataset with many columns, do this in a for loop
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_pres = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# we mainly use two options todeal with categorical features
# hash_bucket and vocabulary_list
# vocabulary_list: we know that for Group, there are 4 possible values - A,B,C,D 
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
# 'Group' is the key followed by the list (i.e. vocabulary list)
# vocabulary_list is useful when there are few well defined categories, like in above case 4
# if we had 100s of these, like countries in the world, above method is not good. Then we use hash_bucket
# so if we are in a situation we we dont know all the groupings or we just dont want to
# type them all manually, use a hash_bucket
#assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group',hash_bucket_size=10)
# hash_bucket_size is the maximum number of groups in that column

# we will now convert a continuous column to categorical column. This is also called Feature Engineering
# so will now convert the Age to a categorical column. 
# we will first try to visualize the columns we are going to categorize
diabetes.Age.hist(bins=20)
age_bucket = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])

# now let us put it all together
feat_cols = [num_preg,plasma_gluc,dias_pres,tricep,insulin,bmi,
             diabetes_pedigree,assigned_group,age_bucket]

# train test split
x_data = diabetes.drop('Class',axis=1)
labels = diabetes['Class']
x_train, x_test, y_train, y_test = train_test_split(x_data,labels,
                                                        test_size=0.3, 
                                                        random_state=101)

# now lets create input functions
# we will be using pandas input functions since we have the data as pandas dataframe
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,
                                                 batch_size=10,
                                                 num_epochs=1000,
                                                 shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)

#now lets train the model
model.train(input_fn=input_func,steps=1000)

# now lets evaluate the model
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,y=y_test,
                                                 batch_size=10,
                                                 num_epochs=1,
                                                 shuffle=False)
# shuffle is false to make sure we are evaluating it in the same order

results = model.evaluate(eval_input_func)

print(results)

# we are getting around 74% accuracy which is not so bad
# lets get some predictions out of it
# since we dont have any data for prediction, we will reuse x_test
# note that we are NOT supplying y_test, otherwise it will become
# an evaluation and not a prediction
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,
                                                 batch_size=10,
                                                 num_epochs=1,
                                                 shuffle=False)
predictions = model.predict(pred_input_func)

# since predictions is going to be a generator
#print(list(predictions))

#### the above was a Linear classifier. Lets now repeat the process with
# a Dense Neural Network Classifier (DNNClassifier)
# hidden_units defines how many neurons you want and how many layers
# so you provide a list of neurons per layer.
# so if I want a 3 layers with 10 neurons each, I say [10,10,10]
# so thats 10 neurons and its densely connected, meaning every
# neuron is connected to every other neuron in the next layer
#dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)

# note that the above train step will generate an error like so:
# ValueError: Items of feature_columns must be a _DenseColumn. 
# this is because we have a categorical columns in feature_columns
# we need to pass the feature_column into an embedding_column
# THIS IS ONLY FOR DENSE NEURAL NETWORKS
# dimension=4 since ['A','B','C','D']
embedded_group_col = tf.feature_column.embedding_column(assigned_group,dimension=4)
# if the feature_columns have more categorical columns, make an embedded_group_col 
# for each
# then we will replace the categorical columns in feat_cols with these embedded cols
# since we only have assigned_group as categorical column, we only replace that
feat_cols = [num_preg,plasma_gluc,dias_pres,tricep,insulin,bmi,
             diabetes_pedigree,embedded_group_col,age_bucket]
# lets crete input_func again
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,
                                                 batch_size=10,
                                                 num_epochs=1000,
                                                 shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],
                                       feature_columns=feat_cols,n_classes=2)
dnn_model.train(input_func,steps=1000)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,y=y_test,
                                                 batch_size=10,
                                                 num_epochs=1,
                                                 shuffle=False)
results = dnn_model.evaluate(eval_input_func)

print(results)

# we get nearly the same accuracy as with Linear Regressor
# we can retry by adding more neurons and layers.
# since we have only 10 features, we cant add too many neurons (or we will end up overfitting)
# we can retry like
#dnn_model = tf.estimator.DNNClassifier(hidden_units=[20,20,20],
#                                       feature_columns=feat_cols,n_classes=2)
# even with above addition of [20,20,20], the resulting accuracy and auc remained almost same
# so looksl ike we have attained the limits of what we can achieve with this data