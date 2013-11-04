import numpy as np
import gb_variables as gv
import birch_ori
import hrbirch
import time,random,pickle
from collections import Counter
labels_ground = []
normlizer = 1

def run(filename = None,cftree = None,offIni = True):
	f = open(filename,'r')
	lines = []
	for i in xrange(gv.INIDATA_NUM):
		lines.append(f.readline())
	iniData = processRawData(lines)
	gv.Dim = len(iniData[0])
	cftree.initClusting(np.array(iniData))
	if offIni: 
		f.seek(0)
		resetGlobal()
	count = 0
	while True:
		lines = f.readlines(gv.MAX_LIMIT_MEMORY)
		count += len(lines)
		print count
		lines = processRawData(lines)
		if not lines:
			cftree.endStep()
			break
		cftree.insertEntries(lines)
	print cftree
	f.close()
	return cftree

def run_brich_ori(filename,dumpfile,res):
	starttime = time.clock()
	gv.cftree = birch_ori.CFTree()
	run(filename,gv.cftree,True)
	endtime = time.clock()
	elapse = endtime - starttime
	with open(res+dumpfile,'wb') as f:
		pickle.dump(gv.cftree,f)
	return gv.cftree,elapse

def run_hrbrich(filename,dumpfile,res):
	starttime = time.clock()
	gv.cftree = hrbirch.CFTree()
	run(filename,gv.cftree,gv.offInit)
	endtime = time.clock()
	elapse = endtime - starttime
	print 'ealpse',elapse
	with open(res+dumpfile,'wb') as f:
		pickle.dump(gv.cftree,f)
	#res = res.split('\\')
	#tmp = res[-1]
	#res[-1] = 'normal'
	#res.append(tmp)
	#res = ('\\').join(res)
	with open(res+'normalizor.pickle','wb') as f:
		pickle.dump(gv.cftree.Normalizor,f)
	print len(labels_ground)
	with open(res+'groundLabels.pickle','wb') as f:
		pickle.dump(labels_ground,f)
	return gv.cftree,elapse

def resetGlobal():
	global labels_ground,normlizer
	normlizer = 1
	labels_ground = []

def processRawData(lines):
	res = []
	for line in lines:
		tmp = np.array([float(x) for x in line.split(',')])
		#fr = 1#np.array([1000,10,1,100,10,100,100,100,100,1000])
		res.append(tmp[0:-1])#[0:-1]
		global labels_ground
		labels_ground.append(tmp[-1])
	return res

if __name__ == '__main__':
	'''
	import sys
	f = open('run.log','a+')
	tmp = sys.stdout
	sys.stdout = f
	'''

	file = 'usps.data'
	path = r'C:\Users\LiuLiang\Desktop\HCbirch\data\exp'
	res = r'C:\Users\LiuLiang\Desktop\HCbirch\result\\'
	filename = path + '\\' + file
	setPara.setMode(file)
	#treeori,timeori = run_brich_ori(filename,'ori_'+file+'.pickle',res)
	treehr,timehr = run_hrbrich(filename,'hr_'+file+'.pickle',res)
	print len(labels_ground)
	#ev.printinfor(treeori,birch_ori.Entry.d2)
	#ev.printinfor(treehr,hrbirch.Entry.d2)
	#f.close()
	#sys.stdout = tmp

