import numpy as np
import gb_variables as gv
import birch_ori
import hrbirch
import time,random,pickle
from collections import Counter

labels_ground = []
normalizer = 1

def globalClustering_hier(leaves,K,dist):
	nodes = list(leaves)
	while len(nodes) > K:
		#print 'clusters num %d' % len(nodes)
		min_dis = float('inf')
		min_pair = (-1,-1)
		for i in range(len(nodes)):
			for j in range(i+1,len(nodes)):
				d = dist(nodes[i],nodes[j])
				if d < min_dis:
					min_dis = d
					min_pair = (i,j)
		i,j = min_pair
		nodes[i].update(nodes[j])  # note i<j
		del nodes[j]
	return nodes


def globalClustering_hier_single(leaves,K,dist):
	nodes = list(leaves)
	length = len(nodes)
	sets = -1 * np.ones([length],dtype = int)
	dismat = [(dist(nodes[i],nodes[j]) ,i,j) for i in range(length) \
			for j in range(length) if i<j]
	dismat.sort()

	def tree_find(i):
		if sets[i] == -1: return i
		else:  sets[i] = tree_find(sets[i])
		return sets[i]

	for k in range(len(dismat)):
		if length <= K: 
			break
		dis ,i,j = dismat[k]
		tree_i = tree_find(i)
		tree_j = tree_find(j)
		assert sets[tree_i] == -1
		assert sets[tree_j] == -1
		if tree_i != tree_j:
			nodes[tree_i].update(nodes[tree_j])
			sets[tree_j] = tree_i
			length -= 1

	tmp = []
	for i in range(len(nodes)):
		if sets[i] == -1: 
			tmp.append(nodes[i])
	assert len(tmp) == K
	return tmp

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
	if leafNodes[0].child == None: return
	for leaf in leafNodes:
		leaf.leafId = 1
		for e in leaf.child.entries:
			leaf.objectIdx.extend(e.objectIdx)

def purity(pred,truth):
	try:
		assert len(pred) == len(truth)
	except:
		import sys
		print >>sys.stderr,len(pred),len(truth)
		assert 1==2
	truth = np.array(truth)
	cnt = 0
	for e in set(pred):
		cnt += max(Counter(truth[pred == e]).values())
	return (cnt+0.0)/len(truth)

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
		ssq += sum(getSquareSum(e)/normalizer)
	return np.sqrt(ssq)

def connectSubCluster(subClu,leafNodes):
	n = 0
	for leaf in leafNodes:
		n += len(leaf.objectIdx)
	labels = -1 * np.ones(n,dtype = int)
	for k in range(len(leafNodes)):
		for idx in leafNodes[k].objectIdx:
			labels[idx-1] = k+1
	print 'len labels',len(labels)
	assert all(labels != -1)
	return labels

def getSquareSum(ety):
	return ety.lss - ety.lls**2/ety.N

def cosis(leaves):
	for e in leaves:
		print e.N,len(e.objectIdx)

def printinfor(tree,disfun,labeled):
	print 'tree summary'
	print tree.__str__()
	#cosis(tree.leaves)
	print 'leaves:'
	count = 0
	for e in tree.leafNodes:
		print e.N,
		count += e.N
	print '\nobject count: ',count
	updateLeafIdx(tree.leafNodes)
	try:
		tree.leafNodes.extend(tree.fruit)
		for e in tree.fruit:
			print 'ss e.n',e.N
	except:
		print 'error in: fruit'
		pass
	print 'length of leafnodes: ',len(tree.leafNodes)
	#cosis(tree.leafNodes)
	print 'start globalcluster complete'
	nodes = globalClustering_hier(tree.leafNodes,gv.NUM_CLUSTERS,disfun)
	print 'nodes length',len(nodes)
	print 'NSSQ_hier: ',NSSQ_hier(nodes)
	if labeled:
		labels_pred = connectSubCluster(range(1,len(nodes)+1),nodes)
		global labels_ground
		print 'purity_comp: ',purity(labels_pred,labels_ground)
	print '\n'

def printinfo_single(tree,disfun,labeled):
	updateLeafIdx(tree.leafNodes)
	try:
		tree.leafNodes.extend(tree.fruit)
	except:
		print 'error in: fruit'
		pass
	print 'start globalcluster single'
	nodes = globalClustering_hier_single(tree.leafNodes,gv.NUM_CLUSTERS,disfun)
	print 'NSSQ_hier: ',NSSQ_hier(nodes)
	if labeled:
		labels_pred = connectSubCluster(range(1,len(nodes)+1),nodes)
		print 'purity_sing: ',purity(labels_pred,labels_ground)
	print '\n'

def setNormAndGnd(norm,gnd):
	global normalizer,labels_ground
	normalizer = norm
	labels_ground = gnd


if __name__ == '__main__':
	import sys
	f = open('eva.log','a+')
	tmp = sys.stdout
	#sys.stdout = f

	file = 'usps.data'
	res = r'C:\Users\LiuLiang\Desktop\HCbirch\result\\'
	with open(res + 'groundLabels.pickle','rb') as f:
		global labels_ground
		labels_ground = pickle.load(f)
		print len(labels_ground)
	with open(res + 'normalizor.pickle','rb') as f:
		global normalizer
		normalizer = pickle.load(f)
	#with open(res + 'ori_'+ file + '.pickle' ,'rb') as f:
	#	ori_tree = pickle.load(f)
	#	printinfor(ori_tree,birch_ori.Entry.d2)
	with open(res + 'hr_' + file +'.pickle' ,'rb') as f:
		gv.cftree = pickle.load(f)
		print 'upperset'
		count = 0
		for e in gv.cftree.upperSet:
			count +=e.N
			print e.N
		print count
		printinfor(gv.cftree,hrbirch.Entry.d2)

	f.close()
	sys.stdout = tmp

