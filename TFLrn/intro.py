# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:50:37 2019

@author: DevAccessa
"""

import tensorflow as tf
import numpy as np
import pandas as pd

#a = tf.constant(2.0,tf.float32)
#b = tf.constant(3.0,tf.float32)
#
#c = a*b
#
#sess = tf.Session()
#print(sess.run(a))
#print(sess.run(b))
#print(sess.run(c))

#sess = tf.Session()
#zero = tf.Variable(0)
#one = tf.constant(1)
#new_value = tf.add(zero,one)
#update = tf.assign(zero,new_value)
##update_constant = tf.assign(one,new_value)
#init_op = tf.global_variables_initializer()
#sess.run(init_op)
#print(sess.run(zero))
#
#for _ in range(5):
#    sess.run(update)
#    print(sess.run(zero))
#    
#sess.close()

#graph = tf.get_default_graph()
#print(graph.get_operations())
#
#a = tf.constant(10,name="a")
#
#
#b = tf.constant(20,name="b")
#
#
#c = tf.add(a,b,name="c")
#
#d = tf.multiply(a,b,name="d")
#
#e = tf.multiply(c,d,name="e")
#sess = tf.Session()
#print(sess.run(e))
##print(graph.get_operations())
#for op in graph.get_operations():
#    print(op.name)
#    
#sess.close()

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data,columns=housing.feature_names)
