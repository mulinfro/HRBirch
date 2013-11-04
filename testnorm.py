import os,pickle
path = r'C:\Users\LiuLiang\Desktop\HCbirch\result2\normal'
for file in os.listdir(path):
	if os.path.isdir(file): continue
	print file
	f = open(os.path.join(path,file),'rb')
	norm = pickle.load(f)
	print len(norm)
	print norm
	print '\n'
