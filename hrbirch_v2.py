import numpy as np

leafDict = {}
LEAF_ID = 0
class Entry(object):
	def __init__(self,ls=0,ss=0,id=-1):
		self.N = 0
		self.lls = ls
		self.lss = ss
		self.child = None
		self.leafId = id
		self.notSelDim = np.zeros(D,dtype = 'bool')
		#self.uls = 0
		#self.uss = 0


	def d0(self,node):
		idx = np.where(not(self.notSelDim or node.notSelDim))[0]
		fractor = Dim/(len(idx)+0.0)
		op1 = self.lls[idx]
		op2 = node.lls[idx]
		rs = (self.lls/self.N - node.lls/node.N)**2
		return fractor*np.sqrt(np.sum(rs))

	def d1(self,node):
		idx = np.where(not(self.notSelDim or node.notSelDim))[0]
		fractor = Dim/(len(idx)+0.0)
		op1 = self.lls[idx]
		op2 = node.lls[idx]
		rs = np.abs(op1/self.N - op2/node.N)
		return fractor*np.sum(rs)

	def d2(self,node):
		idx = np.where(not(self.notSelDim or node.notSelDim))[0]
		fractor = Dim/(len(idx)+0.0)
		op1 = self.lss[idx]
		op2 = node.lss[idx]
		op =self.lls[idx] * node.lls[idx]
		rs = -2*np.sum(op)/(self.N*node.N) + np.sum(op1)/self.N\
				+np.sum(op2)/node.N
		return fractor*np.sqrt(rs)
		
	def d3(self,node):
		idx = np.where(not(self.notSelDim or node.notSelDim))[0]
		fractor = Dim/(len(idx)+0.0)
		ls = self.lls[idx] + node.lls[idx]
		ss = self.lss[idx] + node.lss[idx]
		rs = 2*(self.N + node.N)*sum(ss) - 2*sum(ls**2)
		return fractor*np.sqrt(rs/((self.N + node.N)*(self.N + node.N-1)))


	def d4(self,node):
		idx = np.where(not(self.notSelDim or node.notSelDim))[0]
		fractor = Dim/(len(idx)+0.0)
		f = lambda x,y,n:(np.sum(x)- np.sum(y**2)/n)
		ls = self.lls[idx] + node.lls[idx]
		ss = self.lss[idx] + node.lss[idx]
		return fractor*(f(ss,ls,self.N+node.N) - f(self.lss[idx],self.lls[idx],self.N) \
				- f(node.lss[idx],node.lls[idx],node.N))


	distance = d2

	def radiu(self):
		return np.sqrt(self.lss/self.N - self.lls**2/(self.N**2))

	def update(self,node):
		self.lls += node.lls
		self.lss += node.lss
		self.N += node.N
		self.childList.extend(node.childList)


class Node(Entry):
	def __init__(self):
		entries = []
		super(Node,self).__init__()

	def add(self,ety):
		entries.append(ety)

	def findClosestEntry(self,ety):
		minList = [(entries[i].distance(ety),i) for i in range(len(entries)]
		minEty = min(minList)
		secMin = min(minList.remove(minEty))
		flag = (minEty[0] / secMin[0]) > UNCERTAIN_THRESHOLD
		return (minEty[0],minEty[1],flag)


	def findFarthestEntry(self,ety):
		maxList = [child.distance(ety) for child in childList]
		return max(maxList)

	def findFarthestEntryPair(self):
		if len(entries) < 2: return None
		diameter,(c1,c2) = max([(entries[i].distance(entries[j]),(entries[i],entries[j]))\
				for i,j in xrange(len(entries)) if i<j])
		return (c1,c2)

	def findClosestEntryPair(self):
		if len(entries) < 2: return None
		diameter,(c1,c2) = min([(entries[i].distance(entries[j]),(entries[i],entries[j]))\
				for i,j in xrange(len(entries)) if i<j])
		return (c1,c2)

	def insertEntry(self,ety):
		if  len(self.entries) == 0:
			self.add(ety)
			return (True,False)

		
		dis,closet,uncertain = self.findClosestEntry(ety)
		if uncertain: 
			global cftree
			cftree.upperSet.append(ety)
			return (True,True)
		if closet.child != None:
			dontSplit, uncertain = closet.insertEntry(ety)
			if uncertain: return (True,True)
			if dontSplit:
				closet.update(ety)
				return (True,False)
			elif:
				splitPair = split(closet)
				if len(entries) > MAX_NODE_ENTRIES:
					return (False,False)
				elif applyRefinement:
					merginRefinement(splitPair)
					return True
		elif dis < disThreshold:
			closet.update(ety)
			return (True,False)
		elif len(entries) < MAX_NODE_ENTRIES:
			entries.add(ety)
			return (True,False)
		elif:
			entries.add(ety)
			return (False,False)

	def split(self,closet):
		fe1,fe2 = findFarthestPair(closet.child)
		newety1 = Entry()
		newnode1 = Node()
		newety1.child = newnode1
		newety2 = Entry()
		newnode2 = Node()
		newety2.child = newnode2

		if closet.child.leafid > -1:
			newnode1.leafId = closet.child.leafid
			leafDict[newnode1.leafId] = newnode1
			closet.child.leafid = -1

			LEAF_ID ++
			newnode2.leafId = LEAF_ID
			leafDict[newnode2.leafId] = newnode2


		for e in closet.child.entries:
			dis1 = fe1.distance(e)
			dis2 = fe2.distance(e)
			if dis1 < dis2:
				newety1.update(e)
				newnode1.entries.append(e)
			else:
				newety2.update(e)
				newnode2.entries.append(e)

		self.entries.remove(closet)
		self.entries.extend([newety1,newety2])

		return newety1,newety2


'''
class Leaf(Entry):
	def __init__(self):
		super(Leaf,self).__init__()
'''


class CFTree():
	def __init__():
		self.root = None
		self.leaves = []
		self.normlizor = None
		self.threshold = None
		self.objectNum = 0
		self.leafNum = 0

	def initClusting(self,iniData = None):
		from sklearn.cluster import Kmeans	
		# init-> initialize centers[k-means++, random]   n_init -> number of iterations  n_clusters -> number of cluster
		kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
		idx = kmeans.fit_predict(iniData)
		fs = lambda ss,ls,n:(ss/n-ls**2/n/n)
		f = lambda ss,ls,n,d:np.sqrt(np.sum(fs(ss,ls,n))/d)
		for i in np.unique(idx):
			iniClusterLS[i,:] = sum(iniData[idx ==i], axis =0)
			iniClusterSS[i,:] = sum(iniData[idx ==i]**2, axis =0)
			var.append( (f(iniClusterSS[i,:],iniClusterLS[i,:],nPoi,dDim),i))

		var.sort()
		self.root = Node(id=1)
		for i in range(iniClustersNum):
			ety = Entry(iniClusterLS[var[i][1],:],iniClusterSS[var[i][1],:])
			self.root.add(ety)
			sls += iniData[idx ==var[i][1]]
			sss += iniData[idx ==var[i][1]]**2
			sn += sum(idx ==var[i][1])

		self.normlizor = fs(sss,sls,sn)
		self.threshold = root.findClosetEntries()

	def insertEntry(self, ety):
		self.leafNum += ety.N
		self.root.insertEntry(ety)



def processRawData(lines):
	pass

def hrtree(filename = None):
	f = open(filename,'r')
	cftree = CFtree()
	cftree.initClusting(iniData)
	while True:
		lines = f.readlines()
		lines = processRawData(lines)
		if not lines:
			break
		for line in lines:
			cftree.insertEntry(Node(line))

	f.close()
	return cftree


if __name__ == '__main__':
	filename = ''
	tree = hrtree(filename)
	globalClusting(tree)

