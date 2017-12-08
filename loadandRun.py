import tensorflow as tf
import numpy as np
import xlrd

DATAFILE = "data.xls"

book = xlrd.open_workbook(DATAFILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)])
n_samples = sheet.nrows - 1 #sheets.nrows returns number of rows

x = [float(pair[0]) for pair in data]
y = [float(pair[1]) for pair in data]

print(x)
print("separator")
print(y)

X = tf.placeholder(tf.float32, name = "numberoffires")
Y = tf.placeholder(tf.float32, name = "numberoftheft")

w = tf.get_variable("weight",shape=[])
b = tf.get_variable("bias", shape =[])

saver = tf.train.Saver()
Y_predict = w * X + b

with tf.Session() as sess:
    saver.restore(sess, "tmp/regression/model.ckpt")
    output = sess.run(Y_predict,feed_dict = {X:x})
    print(output)
