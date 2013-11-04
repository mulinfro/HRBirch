import numpy as np
import gb_variables as gv

class Entry(object):
	def __init__(self,idx,n,ls=0,ss=0,id=-1):
		self.N = n
		self.lls = ls
		self.lss = ss
		self.child = None
		self.leafId = id
		self.notSelDim = np.zeros(gv.Dim,dtype = 'bool')
		self.objectIdx = idx

	def getProcessedLs(self,ety,flag):
		idx = np.where(np.logical_not (self.notSelDim + ety.notSelDim))[0]
		if len(idx)==0:
			idx = range(0,gv.Dim)
		ls1 = self.lls[idx]/gv.cftree.Normalizor[idx]
		ls2 = ety.lls[idx]/gv.cftree.Normalizor[idx]
		if flag=='nead_ss':
			ss1 = self.lss[idx]/(gv.cftree.Normalizor[idx]**2)
			ss2 = ety.lss[idx]/(gv.cftree.Normalizor[idx]**2)
			return (len(idx)+0.0,ls1,ls2,ss1,ss2)
		return (fractor,ls1,ls2)

	def d0(self,ety):
		(fractor,ls1,ls2) = self.getProcessedLs(ety,'not')
		rs = (ls1/self.N - ls2/ety.N)**2
		return np.sqrt(np.sum(rs/fractor))

	def d1(self,ety):
		(fractor,ls1,ls2) = self.getProcessedLs(ety,'not')
		rs = np.abs(ls1/self.N - ls2/ety.N)
		return np.sum(rs/fractor)

	def d2(self,ety):
		(fractor,ls1,ls2,ss1,ss2) = self.getProcessedLs(ety,'nead_ss')
		op = ls1*ls2
		rs = ss1/self.N +ss2/ety.N - 2 *op/self.N/ety.N
		rs =np.sum(rs) 
		rs = 0 if rs < 1.0e-8 else rs
		return np.sqrt(rs/fractor)
		'''
		import math
		if math.isnan(np.sqrt(fractor*np.sum(rs))):
			print fractor*np.sum(rs)
			print 'dis'
			print ls1
			print '-----------------------------------'
			print ls2
			print '-----------------------------------'
			print ss1
			print '-----------------------------------'
			print ss2
			print '-----------------------------------'
			print self.objectIdx,ety.objectIdx
			assert 4==2
		'''

	def d3(self,ety):
		(fractor,ls1,ls2,ss1,ss2) = self.getProcessedLs(ety,'nead_ss')
		ls = ls1 + ls2
		ss = ss1 + ss2
		rs = 2 *(self.N + ety.N)*sum(ss) - 2 *sum(ls**2)
		return np.sqrt(rs/((self.N + ety.N)*(self.N + ety.N-1))/fractor)

	def d4(self,ety):
		(fractor,ls1,ls2,ss1,ss2) = self.getProcessedLs(ety,'nead_ss')
		f = lambda x,y,n:(np.sum(x)- np.sum(y**2)/n)
		ls = ls1 + ls2
		ss = ss1 + ss2
		return (f(ss,ls,self.N+ety.N) - f(ss1,ls1,self.N) \
				- f(ss2,ls2,ety.N))/fractor

	distance = d2

	def getRadius(self):
		res = self.lss/self.N - (self.lls/self.N)**2
		res[res < 1.0e-8] = 0
		return np.sqrt(res) /(gv.cftree.Normalizor)
# may be something wrong, every dimension have normalized, how to selction;
#if not normalized how to selection; small starddivation does not imply good features
	def selectDimension(self):
		if gv.offSelDim or self.N < gv.MIN_CLUSTER_SIZE:
			return True
		rdi = self.getRadius()
		k_thre = sorted(rdi)[gv.K_sel]
		self.notSelDim[rdi<k_thre] = False
		self.notSelDim[rdi>=k_thre] = True

	def update(self,ety):
		self.lls += ety.lls
		self.lss += ety.lss
		self.N += ety.N
		if self.child == None or  self.leafId > -1:
			self.objectIdx.extend(ety.objectIdx)

	def __str__(self):
		return  'entry %d \n %s\n%s'%(self.N,str(self.lls),str(self.lss))

#-------------------------------------------------------
#-------------------------------------------------------

class Node(Entry):
	def __init__(self):
		self.entries = []

	def add(self,ety):
		#if len(self.entries)==0 or self.entries[0].leafId > -1:
		gv.LEAF_ID += 1
		ety.leafId = gv.LEAF_ID
		gv.cftree.leaves.append(ety)
		self.entries.append(ety)

	def findClosestEntry(self,ety):
		minList = [ e.distance(ety) for e in self.entries]
		minList = zip(minList,range(len(minList)))
		minEty = min(minList)
		if gv.offUpset or len(minList) == 1: return (minEty[0],self.entries[minEty[1]],False)
		minList.remove(minEty)
		secMin = min(minList)
		if secMin[0] < 1.0e-6:
			flag = True
		else:
			flag = (minEty[0] / secMin[0]) > gv.UNCERTAIN_THRESHOLD
		return (minEty[0],self.entries[minEty[1]],flag)

	def findFarthestEntry(self,ety):
		maxList = [e.distance(ety) for e in self.entries]
		return max(maxList)

	def findFarthestEntryPair(self):
		if len(self.entries) < 2: return (0,None,None)
		lr = xrange(len(self.entries))
		lr = [(i,j) for i in lr for j in lr if i<j]
		diameter,(c1,c2) = max([(self.entries[i].distance(self.entries[j]),(self.entries[i],self.entries[j]))\
				for i,j in lr])
		return (diameter,c1,c2)

	def findClosestEntryPair(self):
		if len(self.entries) < 2: return (0,None,None)
		lr = xrange(len(self.entries))
		lr = [(i,j) for i in lr for j in lr if i<j]
		diameter,(c1,c2) = min([(self.entries[i].distance(self.entries[j]),(self.entries[i],self.entries[j]))\
				for i,j in lr])
		return (diameter,c1,c2)

	def insertEntry(self,ety):
		if  len(self.entries) == 0:
			self.add(ety)
			return (True,False)
		dis,closet,uncertain = self.findClosestEntry(ety)
		if uncertain:
			gv.cftree.upperSet.append(ety)
			if len(gv.cftree.upperSet) > gv.MAX_UPPERSET_CAPACITY: 
				gv.cftree.reallocUpperSet()
			return (True,True)
		if closet.child != None:
			dontSplit, uncertain = closet.child.insertEntry(ety)
			if uncertain: return (True,True)
			if dontSplit:
				closet.update(ety)
				closet.selectDimension()
				return (True,False)
			else:
				splitPair = self.split(closet)
				if gv.cftree.ifReachTreeLimit():
					return (True,False)
				if len(self.entries) > gv.MAX_ENTRIES_NUM:
					return (False,False)
				elif gv.applyRefinement:
					merginRefinement(splitPair)
					return (True,False)
				else:
					return (True,False)
		elif dis <= gv.cftree.threshold:
			closet.update(ety)
			closet.selectDimension()
			return (True,False)
		elif len(self.entries) < gv.MAX_ENTRIES_NUM:
			self.add(ety)
			return (True,False)
		else:
			self.add(ety)
			return (False,False)

		return None

	def newsplit(self,ety):
		newety = Entry(n=0,idx=[])
		newnode = Node()
		newety.child = newnode
		newnode.entries.append(ety)
		newety.update(ety)
		self.entries.append(newety)
		return newety

	def split(self,closet):
		#print 'split'
		dis, fe1,fe2 = closet.child.findFarthestEntryPair()
		newety1 = self.newsplit(fe1)
		newety2 = self.newsplit(fe2)
		closet.child.entries.remove(fe1)
		closet.child.entries.remove(fe2)
		#strategy 1: using d2
		#strategy 1: using d1
		for e in closet.child.entries:
			dis1 = fe1.distance(e)
			dis2 = fe2.distance(e)
			if dis1 < dis2:
				newety1.update(e)
				newety1.child.entries.append(e)
			else:
				newety2.update(e)
				newety2.child.entries.append(e)

		gv.cftree.nodeNum += 1
		gv.cftree.splitNum += 1
		newety1.selectDimension()
		newety2.selectDimension()
		self.entries.remove(closet)
		return newety1,newety2

#-------------------------------------------------------
#-------------------------------------------------------

class CFTree():
	def __init__(self):
		self.root = Node()
		self.leaves = []
		self.threshold = 0
		self.objectNum = 0
		self.nodeNum = 1
		self.upperSet = []
		self.outliers = []
		self.leafNum = 0
		self.rebuildNum  = 0
		self.splitNum = 0
		self.leafNodes = []
		self.Normalizor = None

	def endStep(self):
		if not gv.offUpset:
			#self.threshold = self.threshold * 0.8
			#self.rebuildTree()
			self.moveUppset()
			self.reallcoOutliers()
		self.getLeafNodes(self.root)

	def __str__(self):
		return 'leaves: %d; threshold: %f; objectNum: %d; nodeNum: %d; upperSet: %d; outLiers: %d; rebuildNum: %d; splitNum: %d'\
				%(len(self.leaves),self.threshold,self.objectNum,self.nodeNum,len(self.upperSet),len(self.outliers),self.rebuildNum,self.splitNum)

	def updateNormalizer(self):
		if gv.offNormal: return None
		tls,tss,tn =0,0,0
		for e in self.root.entries:
			tls += e.lls
			tss += e.lss
			tn += e.N
		res = tss/tn - (tls/tn)**2
		res[res < 1.0e-6] = 1
		gv.cftree.Normalizor = np.sqrt(res)

	def reallocUpperSet(self):
		print 'reallocUpperSet'
		tmp = self.upperSet
		self.upperSet = []
		tmp_thr = gv.UNCERTAIN_THRESHOLD
		gv.UNCERTAIN_THRESHOLD = 0.95 
		gv.MAX_UPPERSET_CAPACITY += 1
		for e in tmp:
			self.insertSingleEntry(e)
		gv.MAX_UPPERSET_CAPACITY -= 1
		gv.UNCERTAIN_THRESHOLD = tmp_thr 
		if len(self.upperSet) > gv.OUTLIER_THRES:
			self.moveUppset()
			if len(self.outliers) > gv.OUTLIERS_NUM:
				#self.threshold = self.threshold * 0.9
				#print 'thr ---------',self.threshold 
				#self.rebuildTree()
				self.reallcoOutliers()
		print len(self.upperSet),'||',len(tmp)

	def moveUppset(self):
		self.outliers.extend(self.upperSet)
		self.upperSet = []

	def reallcoOutliers(self):
		print 'reallcoOutliers'
		#self.rebuildTree()
		tmp_thr = gv.UNCERTAIN_THRESHOLD
		gv.UNCERTAIN_THRESHOLD = 1
		tmp = self.outliers
		self.outliers = []
		for e in tmp:
			self.insertSingleEntry(e)
		gv.UNCERTAIN_THRESHOLD = tmp_thr 

	def updateThreshold(self):
		self.leafNum = 0
		thr = self.traverseTree(self.root) / self.leafNum
		#self.threshold  = thr
		self.threshold = 1.1*self.threshold if thr <= self.threshold else thr
		print thr,'---',self.threshold 

	def traverseTree(self,rt):
		if rt.entries[0].child == None:	
			dis,c1,c2 = rt.findClosestEntryPair()
			if dis > 0: self.leafNum += 1
			return dis
		else:
			dis = 0
			for e in rt.entries:
				dis += self.traverseTree(e.child)
			return dis

	def getLeafNodes(self,root):
		for e in root.entries:
			if e.child == None: assert 1==2
			if e.child.entries[0].child == None:
				self.leafNodes.append(e)
			else:
				self.getLeafNodes(e.child)

	def ifReachTreeLimit(self):
		if self.nodeNum > gv.MAX_NODES:
			self.updateThreshold()
			#self.threshold = 0   #  fix no1
			self.rebuildTree()
			return True
		return False

	def rebuildTree(self):
		print 'rebuildTree'
		self.rebuildNum += 1
		self.root = Node()
		tmp = self.leaves
		self.leaves = []
		gv.LEAF_ID = 0
		self.nodeNum = 1
		for e in tmp:
			e.leafId = -1
			self.insertSingleEntry(e)

	def printCFtree(self):
		queue = []
		queue.append(self.root)
		flog = open('D:\\logTree.txt','w')
		while len(queue) > 0:
			node = queue.pop(0)
			ls_str,ss_str,n_str = '','',''
			for e in node.entries:
				tmp = ','.join(['%f,'% a for a in e.lls])
				ls_str += tmp
				ss_str += ','.join(['%f'% a for a in e.lss])
				tmp = 'a'
				n_str += '%d %s'%(e.N,' '*len(tmp))
				if e.child != None: queue.append(e.child)
			flog.writelines([n_str,'\n'])#,'\n',ls_str,'\n',ss_str,'\n\n'])
		flog.write('%d,%d,%d'%(self.objectNum,len(self.root.entries),self.threshold))
		flog.close()

	def initClusting(self,iniData = None):
		if gv.offNormal:
			self.Normalizor = np.ones(gv.Dim,dtype = float)
			tmp = iniData
		else:
			ls = np.sum(iniData,axis=0)
			ss = np.sum(iniData**2,axis=0)
			res = ss/len(iniData) -  (ls/len(iniData))**2
			res[res < 1.0e-8] = 1
			gv.cftree.Normalizor = np.sqrt(res)
			tmp = iniData/gv.cftree.Normalizor
		if gv.offInit: return None
		from sklearn.cluster import KMeans
		# init-> initialize centers[k-means++, random]   n_init -> number of iterations  n_clusters -> number of cluster
		kmeans = KMeans(init='k-means++', n_clusters= 5 *gv.NUM_CLUSTERS, n_init=10)
		idx = kmeans.fit_predict(tmp)
		gv.Dim = len(iniData[0])
		for i in np.unique(idx):
			ils = np.sum(iniData[idx ==i,:], axis =0)
			iss = np.sum(iniData[idx ==i,:]**2, axis =0)
			iidx = np.where(idx==i)[0].tolist()
			ety = Entry(ls = ils, ss = iss,n = sum(idx ==i ), idx= iidx)
			self.root.add(ety)

		self.root.entries.sort(key = lambda x: -x.N)
		vol = self.root.entries[0].N * 1/4
		print vol
		while True:
			li = self.root.entries.pop()
			if li.N > vol:
				self.root.entries.append(li)
				break
			else:
				for i in  li.objectIdx:
					self.upperSet.append(Entry(ls = iniData[i-1],ss = iniData[i-1]**2,n=1,idx = [i]))

		for e in self.root.entries:
			print 'root',e.N
			gv.LEAF_ID += 1
			e.leafId = gv.LEAF_ID
			self.leaves.append(e)
			e.selectDimension()

		radiu = [np.sum(e.getRadius()) for e in self.root.entries]
		self.threshold = 1.0/3*min(radiu)
		self.objectNum = len(iniData)

		while len(self.root.entries) > gv.MAX_ENTRIES_NUM:
			self.splitRoot()
			#self.printCFtree()
		print self

	def splitRoot(self):
		print 'split root'
		tmp = self.root
		self.root = Node()
		dis, fe1,fe2 = tmp.findFarthestEntryPair()
		newety1 = self.root.newsplit(fe1)
		newety2 = self.root.newsplit(fe2)
		tmp.entries.remove(fe1)
		tmp.entries.remove(fe2)
		for e in tmp.entries:
			dis1 = fe1.distance(e)
			dis2 = fe2.distance(e)
			if dis1 < dis2:
				newety1.update(e)
				newety1.child.entries.append(e)
			else:
				newety2.update(e)
				newety2.child.entries.append(e)

		newety1.selectDimension()
		newety2.selectDimension()
		self.nodeNum += 1
		self.splitNum += 1
		while True:
			doSplit = False
			for e in self.root.entries:
				if e.child != None and len(e.child.entries) > gv.MAX_ENTRIES_NUM:
					self.root.split(e)
					doSplit = True
					break
			if not doSplit:
				break

	def insertSingleEntry(self, ety):
		f1,f2 = self.root.insertEntry(ety)
		if f1 == False and f2 == False:
			#print 'insert splitRoot'
			self.splitRoot()

	def insertEntries(self, lines):
		for line in lines:
			self.objectNum += 1
			if self.objectNum % gv.UPDATE_NORMAL_STEP ==0: self.updateNormalizer()
			self.insertSingleEntry(Entry(ls=line,ss=line**2,n=1,idx = [self.objectNum]))
#------------------------------------------------------
#------------------------------------------------------

def processRawData(lines):
	res = []
	for line in lines:
		tmp = np.array([float(x) for x in line.split(',')])
		print tmp
		assert 1==2
		res.append(tmp[0:10])

	return res

def run(filename = None,cftree = None):
	f = open(filename,'r')
	lines = []
	for i in xrange(gv.INIDATA_NUM):
		lines.append(f.readline())
	iniData = processRawData(lines)
	cftree.initClusting(np.array(iniData))
	count = 0
	while True:
		lines = f.readlines(gv.MAX_LIMIT_MEMORY)
		#print len(lines)
		lines = processRawData(lines)
		count += 1
		print count
		if not lines:
			break
		if count ==4:
			cftree.printCFtree()
			print len(cftree.leafNodes)
			for e in cftree.leafNodes:
				print e
			break

		cftree.insertEntries(lines)
		cftree.getLeafNodes(cftree.root)
		#cftree.printCFtree()
	f.close()
	return cftree

if __name__ == '__main__':
	filename = r'C:\Users\LiuLiang\Desktop\HCbirch\data\exp\kddcup_10_percent.data'
	gv.cftree = CFTree()
	run(filename,gv.cftree)
	#globalClusting(gv.cftree)
# how to updata radius threshold or using number of entries replace

