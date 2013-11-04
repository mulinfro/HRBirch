import numpy as np
import gb_variables as gv

class Entry(object):
	def __init__(self,idx,n,ls=0,ss=0,id=-1):
		self.N = n
		self.lls = ls
		self.lss = ss
		self.child = None
		self.leafId = id
		self.objectIdx = idx

	def getProcessedLs(self,ety,flag):
		if flag=='nead_ss':
			return (self.lls,ety.lls,self.lss,ety.lss)

		return (self.lls,ety.lls)

	def d0(self,ety):
		(ls1,ls2) = self.getProcessedLs(ety,'not')
		rs = (ls1/self.N - ls2/ety.N)**2
		return np.sqrt(np.sum(rs))

	def d1(self,ety):
		(ls1,ls2) = self.getProcessedLs(ety,'not')
		rs = np.abs(ls1/self.N - ls2/ety.N)
		return np.sum(rs)

	def d2(self,ety):
		(ls1,ls2,ss1,ss2) = self.getProcessedLs(ety,'nead_ss')
		op = ls1*ls2
		rs =  ss1/self.N +ss2/ety.N - 2 *op/self.N/ety.N
		return np.sqrt(np.sum(rs))

	def d3(self,ety):
		(ls1,ls2,ss1,ss2) = self.getProcessedLs(ety,'nead_ss')
		ls = ls1 + ls2
		ss = ss1 + ss2
		rs = 2 *(self.N + ety.N)*sum(ss) - 2*sum(ls**2)
		return np.sqrt(rs/((self.N + ety.N)*(self.N + ety.N-1)))


	def d4(self,ety):
		(ls1,ls2,ss1,ss2) = self.getProcessedLs(ety,'nead_ss')
		f = lambda x,y,n:(np.sum(x)- np.sum(y**2)/n)
		ls = ls1 + ls2
		ss = ss1 + ss2
		return f(ss,ls,self.N+ety.N) - f(ss1,ls1,self.N) \
				- f(ss2,ls2,ety.N)


	distance = d2

	def getRadius(self):
		return np.sqrt((self.lss/self.N - (self.lls/self.N)**2))

	def update(self,ety):
		self.lls += ety.lls
		self.lss += ety.lss
		self.N += ety.N
		if self.child == None or self.leafId > -1:
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
		return (minEty[0],self.entries[minEty[1]])


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
			return True
		dis,closet = self.findClosestEntry(ety)
		if closet.child != None:
			assert closet.child != closet
			dontSplit = closet.child.insertEntry(ety)
			if dontSplit:
				closet.update(ety)
				return True
			else:
				splitPair = self.split(closet)
				gv.cftree.ifReachTreeLimit()
				return len(self.entries) <= gv.MAX_ENTRIES_NUM
		elif dis <= gv.cftree.threshold:
			closet.update(ety)
			return True
		else:
			self.add(ety)
			return len(self.entries) <= gv.MAX_ENTRIES_NUM

	def newsplit(self,ety):
		newety = Entry(n=0,idx = [])
		newnode = Node()
		newety.child = newnode
		newnode.entries.append(ety)
		newety.update(ety)
		self.entries.append(newety)
		return newety

	def split(self,closet):
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
		self.entries.remove(closet)
		return newety1,newety2


#-------------------------------------------------------
#-------------------------------------------------------

class CFTree():
	def __init__(self):
		self.root = Node()
		self.leaves = []
		self.nodeNum = 1
		self.threshold = 0
		self.objectNum = 0
		self.leafNum = 0
		self.rebuildNum  = 0
		self.splitNum = 0
		self.leafNodes = []
	def endStep(self):
		self.getLeafNodes(self.root)

	def initClusting(self,dummy):
		pass

	def __str__(self):
		return 'leaves: %d; threshold: %f; objectNum: %d; nodeNum: %d; rebuildNum: %d; splitNum: %d'\
				%(len(self.leaves),self.threshold,self.objectNum,self.nodeNum,self.rebuildNum,self.splitNum)

	def standard_deviation(self):
		f = lambda ls,ss,n:np.sqrt((ss/n - (ls/n)**2))
		return f

	def updateThreshold(self):
		self.leafNum = 0
		thr = self.traverseTree(self.root) / self.leafNum
		self.threshold = 1.1*self.threshold if thr <= self.threshold else thr
		print thr,'----',self.threshold 

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
			self.rebuildTree()
			return True
		return False

	def rebuildTree(self):
		print 'rebuildTree'
		self.rebuildNum  += 1
		self.root = Node()
		tmp = self.leaves
		self.leaves = []
		gv.LEAF_ID = 0
		self.nodeNum = 1 
		#Entry.distance = Entry.d4
		for e in tmp:
			e.leafId = -1
			self.insertSingleEntry(e)
		#Entry.distance = Entry.d2

	def printCFtree(self):
		queue = []
		queue.append(self.root)
		flog = open('D:\\logTree.txt','w')
		while len(queue) > 0:
			node = queue.pop(0)
			ls_str,ss_str,n_str = '','',''
			for e in node.entries:
				tmp = ','.join([ '%f,'% a for a in e.lls])
				ls_str += tmp
				ss_str += ','.join([ '%f'% a for a in e.lss])
				tmp = 'a'
				n_str += '%d %s'%(e.N,' '*len(tmp))
				if e.child != None: queue.append(e.child)

			flog.writelines([n_str,'\n'])#,'\n',ls_str,'\n',ss_str,'\n\n'])
		flog.write('%d,%d,%d'%(self.objectNum,len(self.root.entries),self.threshold))
		flog.close()

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

		self.splitNum += 1
		self.nodeNum += 1

	def insertSingleEntry(self, ety):
		f = self.root.insertEntry(ety)
		if f == False:
			#print 'insert splitRoot'
			self.splitRoot()

	def insertEntries(self, lines):
		for line in lines:
			self.objectNum += 1
			self.insertSingleEntry(Entry(ls=line,ss=line**2,n=1,idx = [self.objectNum]))

#------------------------------------------------------
#------------------------------------------------------

def processRawData(lines):
	res = []
	for line in lines:
		tmp = np.array([float(x) for x in line.split(',')])
		fr = np.array([1000,10,1,100,10,100,100,100,100,1000])
		res.append(tmp[0:10]/fr)

	return res

def run(filename = None,cftree = None):
	f = open(filename,'r')
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
			return

		cftree.insertEntries(lines)
		#cftree.printCFtree()

	f.close()
	return cftree


if __name__ == '__main__':
	filename = r'C:\Users\LiuLiang\Desktop\HCbirch\data\covtype.data'
	gv.cftree = CFTree()
	run(filename,gv.cftree)
	#globalClusting(gv.cftree)


# how to updata radius threshold or using number of entries replace



