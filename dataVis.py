import numpy as np
import matplotlib.pyplot as plt
def processRawData(lines):
	res = []
	res = np.zeros([len(lines),2],dtype = float)
	for i in range(len(lines)):
		line = lines[i]
		res[i,:] = np.array([float(x) for x in line.split(',')])
		#s = (',').join([str(x) for x in tmp])
		#res.append(tmp)
	return res

def run(filename = None):
	f = open(filename,'r')
	lines = f.readlines()
	print len(lines)
	lines = processRawData(lines)
	plt.plot(lines[50000:-1,0],lines[50000:-1,1],'o')
	#plt.plot(lines[1000:2000,0],lines[1000:2000,1],'or')
	#plt.plot(lines[200:300,0],lines[200:300,1],'oy')
	plt.show()


if __name__ == '__main__':
	filename = r'C:\Users\LiuLiang\Desktop\HCbirch\data\exp\birch1.data'
	run(filename)
