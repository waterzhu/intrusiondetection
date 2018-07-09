import csv
import numpy as np
from sklearn.cross_validation import train_test_split


f_train,f_test,l_train,l_test = [],[],[],[]

def split(data):
        x,y = [],[]
        global f_train
        global f_test
        global l_train
        global l_test

        for row in data:
                x.append(row[:-1])
                y.append(row[-1])
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
        f_train = f_train + x_train
        f_test = f_test + x_test
        l_train = l_train + y_train
        l_test = l_test + y_test


with open('train10.csv','r') as csvfile:
        reader = csv.reader(csvfile)
#       feature,label = [], []
        class0,class1,class2,class3,class4,class5,class6,class7 = [],[],[],[],[],[],[],[]
        for row in reader:
                if row[-1] == '0':
                        class0.append(row)
                if row[-1] == '1':
                        class1.append(row)
                if row[-1] == '2':
                        class2.append(row)
                if row[-1] == '7':
                        class7.append(row)
                if row[-1] == '3':
                        class3.append(row)
                if row[-1] == '4':
                        class4.append(row)
                if row[-1] == '5':
                        class5.append(row)
                if row[-1] == '6':
                        class6.append(row)
        print('load data over')
	print(len(class0))
	print(len(class1))
	print(len(class2))
	print(len(class3))
	print(len(class4))
	print(len(class5))
	print(len(class6))
	print(len(class7))
        split(class0)
        split(class1)
	split(class2)
 	split(class3)
        split(class4)
        split(class5)
        split(class6)
        split(class7)
        print('split data over')

class0,class1,class2,class3,class4,class5,class6,class7 = [],[],[],[],[],[],[],[]
print(f_train[0])
print(len(f_train))
for i in range(len(f_train)):
	f_train[i].append(l_train[i])
for i in range(len(f_test)):
	f_test[i].append(l_test[i])

print(f_train[0])

with open('traind10.csv','a') as out:
	writer = csv.writer(out)
	for line in f_train:
		writer.writerow(line)
print('train set over')
with open('test10.csv','a') as out1:
	writer = csv.writer(out1)
	for line in f_test:
		writer.writerow(line)
print('test set over') 
