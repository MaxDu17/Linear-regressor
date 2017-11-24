import numpy as np
import matplotlib as plotter
import tensorflow as tf
import xlrd

DATAFILE = "data.xls"

book = xlrd.open_workbook(DATAFILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1,sheet.nrows)])
n_samples = sheet.nrows - 1 #sheets.nrows returns number of rows

for x, y in data:
    print(x)
   # print("____")
    print(y)
    print("end")
