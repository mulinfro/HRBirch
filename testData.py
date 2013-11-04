import numpy as np
def processRawData(lines):
	res = []
	for line in lines:
		tmp = np.array([float(x) for x in line.split(',')])
		#print tmp[0:-1]
		res.append(tmp[0:-1])
	return res

def run(filename = None):
	f = open(filename,'r')
	lines = []
	count = 0
	sum = 0
	s = set()
	fc = 0
	while True:
		lines = f.readlines(1024*200)
		#print len(lines)
		lines = processRawData(lines)
		count += 1
		for line in lines:
			if line[0] >0: 
				fc+=1
				s.add(line[0])
			sum += line
		print sum[0]
		print count
		if not lines:
			print sum
			print fc
			print s
			break
	f.close()

if __name__ == '__main__':
	filename = r'C:\Users\LiuLiang\Desktop\HCbirch\data\exp\kddcup_10_percent.data'
	run(filename)
	#globalClusting(gv.cftree)
# how to updata radius threshold or using number of entries replace

