import gb_variables as gv
def setValues(file):
	if file == 'usps.data':
		gv.LEAF_ID = 0
		gv.INIDATA_NUM = 250
		gv.UPDATE_NORMAL_STEP = 200
		gv.MAX_ENTRIES_NUM = 4
#MAX_LEAgv.F_NUM = 4
		gv.count = 0
		gv.NUM_CLUSTERS = 10
		gv.Dim = 256
		gv.K_sel = 155
		gv.Normalizor = None
		gv.MAX_LIMIT_MEMORY = 1024*400
		gv.UNCERTAIN_THRESHOLD = 0.9
		gv.MAX_UPPERSET_CAPACITY = 200
		gv.OUTLIER_THRES = 0.8*gv.MAX_UPPERSET_CAPACITY
		gv.OUTLIERS_NUM = 1000
		gv.cftree = None
		gv.applyRefinement = False
		gv.MIN_CLUSTER_SIZE = 20
		gv.MAX_NODES = 50
		gv.ENTRY_LIMIT = 1000
	elif file == 'covtype.data':
		gv.LEAF_ID = 0
		gv.INIDATA_NUM = 5000
		gv.UPDATE_NORMAL_STEP = 2000
		gv.MAX_ENTRIES_NUM = 10
		gv.count = 0
		gv.NUM_CLUSTERS = 7
		gv.Dim = 10
		gv.K_sel = 6
		gv.Normalizor = None
		gv.MAX_LIMIT_MEMORY = 1024*200
		gv.UNCERTAIN_THRESHOLD = 0.9
		gv.MAX_UPPERSET_CAPACITY = 5000
		gv.OUTLIER_THRES = 0.8*gv.MAX_UPPERSET_CAPACITY
		gv.OUTLIERS_NUM = 30000
		gv.cftree = None
		gv.applyRefinement = False
		gv.MIN_CLUSTER_SIZE = 50
		gv.MAX_NODES = 1000
		gv.ENTRY_LIMIT = 30000
	elif file == 'kddcup_10_percent.data':
		gv.LEAF_ID = 0
		gv.INIDATA_NUM = 5000
		gv.UPDATE_NORMAL_STEP = 2000
		gv.MAX_ENTRIES_NUM = 8
		gv.count = 0
		gv.NUM_CLUSTERS = 5
		gv.Dim = 34
		gv.K_sel = 20
		gv.Normalizor = None
		gv.MAX_LIMIT_MEMORY = 1024*400
		gv.UNCERTAIN_THRESHOLD = 0.9
		gv.MAX_UPPERSET_CAPACITY = 5000
		gv.OUTLIER_THRES = 0.8*gv.MAX_UPPERSET_CAPACITY
		gv.OUTLIERS_NUM = 30000
		gv.cftree = None
		gv.applyRefinement = False
		gv.MIN_CLUSTER_SIZE = 50
		gv.MAX_NODES = 800
		gv.ENTRY_LIMIT = 20000
	elif file == 'birch1.data' or file == 'birch2.data' or file == 'birch3.data':
		gv.LEAF_ID = 0
		gv.INIDATA_NUM = 2000
		gv.UPDATE_NORMAL_STEP = 1000
		gv.MAX_ENTRIES_NUM = 8
		#MAX_LEAgv.F_NUM = 4
		gv.count = 0
		gv.NUM_CLUSTERS = 100
		gv.Dim = 2
		gv.K_sel = 2
		gv.Normalizor = None
		gv.MAX_LIMIT_MEMORY = 1024*100
		gv.UNCERTAIN_THRESHOLD = 0.9
		gv.MAX_UPPERSET_CAPACITY = 1000
		gv.OUTLIER_THRES = 0.8*gv.MAX_UPPERSET_CAPACITY
		gv.OUTLIERS_NUM = 6000
		gv.cftree = None
		gv.applyRefinement = False
		gv.MIN_CLUSTER_SIZE = 30
		gv.MAX_NODES = 800
		gv.ENTRY_LIMIT = 2000
	elif file in ['dim256.data', 'dim512.data','dim1024.data']:
		gv.LEAF_ID = 0
		gv.INIDATA_NUM = 300
		gv.UPDATE_NORMAL_STEP = 200
		gv.MAX_ENTRIES_NUM = 3
		#MAX_LEAgv.F_NUM = 4
		gv.count = 0
		gv.NUM_CLUSTERS = 10
		gv.Dim = 256
		gv.K_sel = int(gv.Dim*0.6)
		gv.Normalizor = None
		gv.MAX_LIMIT_MEMORY = 1024*50
		gv.UNCERTAIN_THRESHOLD = 0.9
		gv.MAX_UPPERSET_CAPACITY = 80
		gv.OUTLIER_THRES = 0.8*gv.MAX_UPPERSET_CAPACITY
		gv.OUTLIERS_NUM = 200
		gv.cftree = None
		gv.applyRefinement = False
		gv.MIN_CLUSTER_SIZE = 5
		gv.MAX_NODES = 30
		gv.ENTRY_LIMIT = 150
	else: 
		pass

def setMode(st):
	gv.offInit =  True
	gv.offNormal = True
	gv.offSelDim = True
	gv.offUpset = True
	for m in st:
		if m==1:
			gv.offInit = False
		elif m==2:
			gv.offNormal = False
		elif m==3:
			gv.offSelDim = False
		elif m==4:
			gv.offUpset = False
		else:
			pass
