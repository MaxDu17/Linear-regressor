import numpy as np
import matplotlib as plotter
import tensorflow as tf
import xlrd

DATAFILE = "data.xls"

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

X = tf.placeholder(tf.float32, name = "numberoffires")
Y = tf.placeholder(tf.float32, name = "numberoftheft")

w = tf.Variable(0.0, name = "weight")
b = tf.Variable(0.0, name = "bias")

Y_predict = w * X + b

loss = tf.reduce_mean(tf.square(Y-Y_predict, name = "loss")) #this creates residual of the entire thing
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0035).minimize(loss)
saver = tf.train.Saver()
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('tmp/regression',sess.graph)
    
    for iteration in range(2500):
        sess.run(optimizer, feed_dict = {X:x,Y:y})
        tmploss = sess.run(loss, feed_dict = {X:x,Y:y})
        print(tmploss)
        
            
    writer.close()
    saver.save(sess, "tmp/regression/model.ckpt")
    print(w.eval())
    print(b.eval())
    
