import numpy as np
import matplotlib as plotter
import tensorflow as tf
import xlrd

DATAFILE = "data.xls"

book = xlrd.open_workbook(DATAFILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)])
n_samples = sheets.nrows - 1 #sheets.nrows returns number of rows

X = tf.placeholder(tf.float32, name = "numberoffires")
Y = tf.placeholder(tf.float32, name - "numberoftheft")

w = tf.Variable(0.0, name = "weight")
b = tf.Variable(0.0, name = "bias")

Y_predict = w * X + b

loss = tf.square(Y-Y_predict, name = "loss") #this creates residual of the current point
totalLoss = tf.Variable(0.0, name = "total loss")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(totalLoss)

with tf.Session as sess:
    sess.run(tf.global_variable_initializer())
    writer = tf.summary.fileWriter('tmp/regression',sess.graph)
    
    for iteration in range(100):

        for x, y in data:
            
            sess.run(Y_precict, feed_dict={X:x})
            sess.run(loss, feed_dict = {Y:y})
            totalLoss += loss
            sess.run(optimizer,feed_dict={X:x,Y:y})
