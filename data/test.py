import csv

with open('test.csv','r') as csvfile:
	reader = csv.reader(csvfile)
	label1 = set()
	i,j,k,l,q,w,e,r = 0,0,0,0,0,0,0,0
	for row in reader:
#	print(len(table[1]))
#	table1 = table[1:]
#	feature = [row[:-1] for row in table1]
#	print(feature[0])
#	label = [row[-1] for row in table1]
		
#	for l in label:
#		label1.add(row[-1])
		if row[-1] == '0':
			i = i+1
		if row[-1] == '1':
			r = r+1
#			print(row[-1])
		if row[-1] == '2':
			j = j+1
		if row[-1] == '3':
			k = k+1
		if row[-1] == '4':
			l = l+1
		if row[-1] == '5':
			q = q+1
		if row[-1] == '6':
			w = w+1
		if row[-1] == '7':
			e = e+1

#	print(label1)
	print(i)
	print(r)
	print(j)
	print(k)
	print(l)
	print(q)
	print(w)
	print(e)
# print label
'''
	n = 5
	feature10 = []
	for i in range(len(feature)-n):
		features = []
		for j in range(n):
#			featurex = map(float, feature[i+j])
#			print(featurex)
			features = features + feature[i+j]
		if label[i+n] == 'BENIGN':
			features.append('0')
		else:
			features.append('1')
		feature10.append(features)
	print(feature10[0])
'''
