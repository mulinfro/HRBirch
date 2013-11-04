import numpy as np
def processRawData(lines):
	res = []
	for line in lines:
		tmp = np.array([float(x) for x in line.split()])
		s = (',').join([str(x) for x in tmp]) + ', 1'
		res.append(s+'\n')
	return res

def run(filename = None):
	with open(filename+'P','w') as fout:
		f = open(filename,'r')
		count = 0
		while True:
			lines = f.readlines(1024*600)
			print len(lines)
			lines = processRawData(lines)
			if not lines:
				break
			fout.writelines(lines)

		f.close()


if __name__ == '__main__':
	filename = r'C:\Users\LiuLiang\Desktop\HCbirch\data\cluster dataset\dim256.data'
	run(filename)
	filename = r'C:\Users\LiuLiang\Desktop\HCbirch\data\cluster dataset\dim512.data'
	run(filename)
	filename = r'C:\Users\LiuLiang\Desktop\HCbirch\data\cluster dataset\dim1024.data'
	run(filename)
