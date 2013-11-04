import numpy as np
import gb_variables as gv
import birch_ori
import hrbirch
import time,random,pickle
from collections import Counter
labels_ground = []
normlizer = 1

def globalClustering_hier(leaves,K,dist):
	nodes = list(leaves)
	while len(nodes) > K:
		print 'clusters num %d' % len(nodes)
		min_dis = float('inf')
		min_pair = (-1,-1)
		for i in range(len(nodes)):
			for j in range(i+1,len(nodes)):
				d = dist(nodes[i],nodes[j])
				if d < min_dis:
					min_dis = d
					min_pair = (i,j)
		i,j = min_pair
		# note i<j
		nodes[i].update(nodes[j])
		del nodes[j]
	return nodes

def globalClustering_kmedoids(X,K,dist):
	N = len(X)
	labels = np.zeros(N, dtype = int)
	idx_c = random.sample(range(N), K)

	def dismat(X,Y):
		dis = np.zeros((len(X),len(Y)),dtype = float)
		for i in range(len(X)):
			for j in range(len(Y)):
				dis[i,j] = dist(X[i],Y[j])
		return dis

	dist = dismat(X,X)
	iter = 0
	while True:
		iter += 1
		print 'iter ',iter
		prelabels = labels
		labels = np.argmin(dist[:,idx_c],axis=1)
		for i in range(K):
			idx_i = np.where(labels == i)[0]
			assert idx_i != []
			dist_i = dist[idx_i,idx_i]
			idx = np.argmin(np.sum(dist_i,axis=0))
			try:
				idx_c[i] = idx_i[idx]
			except:
				print i,idx
		if np.all(labels == prelabels):
			break
	return labels


def updateLeafIdx(leafNodes):
	for leaf in leafNodes:
		leaf.LeafId = 1
		for e in leaf.child.entries:
			leaf.objectIdx.extend(e.objectIdx)

def purity(pred,truth):
	assert len(pred) == len(truth)
	pred = np.array(pred)
	truth = np.array(truth)
	cnt = 0
	for e in set(pred):
		cnt += max(Counter(truth[pred == e]).values())
	return (cnt+0.0)/len(truth)

def connectSubCluster(subClu,leafNodes):
	n = 0
	for leaf in leafNodes:
		n += len(leaf.objectIdx)
	labels = -1 * np.ones((n,1),dtype = int)
	for k in range(len(leafNodes)):
		for idx in leafNodes[k].objectIdx:
			labels[idx-1] = subClu[k]
	assert all(labels != -1)
	return labels

def getNormlizer(entries):
	normlizer = birch_ori.Entry()
	for e in entries:
		normlizer.update(e)
	return normlizer

def getSquareSum(ety):
	return ety.lss - ety.lls**2/ety.N

def NSSQ(leafNodes,labels):
	K = max(labels)
	ssq = 0
	clu_i = []
	for i in range(K):
		for j in range(len(leafNodes)):
				if labels[j] == i:
					clu_i.append(leafNodes[j])
		ni = getNormlizer(clu_i)
		ssq += sum(ni.getRadius()**2*ni.N/normlizer)
	return np.sqrt(ssq)

def NSSQ_hier(leafNodes):
	ssq = 0
	for e in leafNodes:
		ssq += sum(getSquareSum(e)/gv.cftree.Normalizor)
	return np.sqrt(ssq)

def run(filename = None,cftree = None,iniCluster = False):
	f = open(filename,'r')
	if iniCluster:
		lines = []
		for i in xrange(gv.INIDATA_NUM):
			lines.append(f.readline())
		iniData = processRawData(lines)
		cftree.initClusting(np.array(iniData))
	count = 0
	while True:
		lines = f.readlines(gv.MAX_LIMIT_MEMORY)
		count += len(lines)
		print count
		lines = processRawData(lines)
		if not lines:
			if iniCluster:
				cftree.moveUppset()
				cftree.reallcoOutliers()
			break
		cftree.insertEntries(lines)
	print cftree
	cftree.getLeafNodes(cftree.root)
	f.close()
	return cftree

def run_brich_ori(filename,dumpfile):
	starttime = time.clock()
	gv.cftree = birch_ori.CFTree()
	run(filename,gv.cftree)
	endtime = time.clock()
	elapse = endtime - starttime
	with open(dumpfile,'wb') as f:
		pickle.dump(gv.cftree,f)
	with open('groundLabels','wb') as f:
		pickle.dump(labels_ground,f)
	return gv.cftree,elapse

def run_hrbrich(filename,dumpfile):
	starttime = time.clock()
	gv.cftree = hrbirch.CFTree()
	run(filename,gv.cftree,True)
	endtime = time.clock()
	elapse = endtime - starttime
	with open(dumpfile,'wb') as f:
		pickle.dump(gv.cftree,f)
	return gv.cftree,elapse

def printinfor(tree,disfun):
	print tree.__str__()
	#nodes = globalClustering_hier(gv.cftree.leaves,7,birch_ori.Entry.d2)
	print 'start globalcluster'
	updateLeafIdx(tree.leafNodes)
	print 'length of leafnodes: ',len(tree.leafNodes)
	nodes = globalClustering_hier(tree.leafNodes,10,disfun)
	print 'end globalcluster'
	print NSSQ_hier(tree.leafNodes)
	labels_pred = connectSubCluster(range(1,len(nodes)+1),nodes)
	global labels_ground
	print purity(labels_pred,labels_ground)
	#print labels

def processRawData(lines):
	res = []
	for line in lines:
		tmp = np.array([float(x) for x in line.split(',')])
		#fr = 1#np.array([1000,10,1,100,10,100,100,100,100,1000])
		res.append(tmp[0:-1])
		global labels_ground
		labels_ground.append(tmp[-1])
	return res

if __name__ == '__main__':
	file = 'usps.data'
	path = r'C:\Users\LiuLiang\Desktop\HCbirch\data'
	res = r'C:\Users\LiuLiang\Desktop\HCbirch\result\\'
	filename = path + '\\' + file
	#treeori,timeori = run_brich_ori(filename,'ori_'+file+'.pickle')
	#treehr,timehr = run_hrbrich(filename,'hr_'+file+'.pickle')
	#printinfor(treeori,birch_ori.Entry.d2)
	with open(res + 'groundLabels','rb') as f:
		labels_ground = pickle.load(f)
	with open(res + 'hr_' + file +'.pickle' ,'rb') as f:
		gv.cftree = pickle.load(f)
	printinfor(gv.cftree,hrbirch.Entry.d2)
	with open(res + 'ori_'+ file + '.pickle' ,'rb') as f:
		ori_tree = pickle.load(f)
	printinfor(ori_tree,birch_ori.Entry.d2)
