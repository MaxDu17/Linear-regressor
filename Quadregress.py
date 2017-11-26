import numpy as np
import matplotlib as plotter
import tensorflow as tf
import xlrd

DATAFILE = "quaddata.xlsx"

book = xlrd.open_workbook(DATAFILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)])
n_samples = sheet.nrows - 1 #sheets.nrows returns number of rows

x = [float(pair[0]) for pair in data]
y = [float(pair[1]) for pair in data]
#x = [1,2,3] 
print(x)
print("separator")
print(y)

X = tf.placeholder(tf.float32, name = "input")
Y = tf.placeholder(tf.float32, name = "output")

a = tf.Variable(0.0, name = "a")
b = tf.Variable(0.0, name = "b")
c = tf.Variable(0.0, name = "c")

Y_predict = a * X*X + b * X + c

#loss = tf.reduce_mean(tf.square(Y-Y_predict, name = "loss")) #this creates residual of the entire thing
loss = tf.losses.huber_loss(Y, Y_predict)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('tmp/regression',sess.graph)
    
    for iteration in range(10000):
        sess.run(optimizer, feed_dict = {X:x,Y:y})
        tmploss = sess.run(loss, feed_dict = {X:x,Y:y})
        print(tmploss)
        
            
    writer.close()
    print(a.eval())
    print(b.eval())
    print(c.eval())
    
    
