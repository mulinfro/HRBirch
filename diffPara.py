import numpy as np
import gb_variables as gv
import birch_ori
import hrbirch
import time,random,pickle
import evalute,setPara,clustering
import sys,os
files = ['birch1.data','dim256.data','dim512.data','dim1024.data']#,'usps.data',''usps.data','kddcup_10_percent.data','covtype.data','birch2.data',
modeState = None
labeled = True
#files = ['birch1.data','birch2.data']#,'birch3.data']#,'kddcup_10_percent.data']#,'usps.data','covtype.data',]
def setRunMode(file):
	global modeState,labeled
	labeled = True
	if file == 'birch1.data' or file == 'birch2.data':
		modeState = [[4]]
	elif file == 'usps.data':
		modeState = [[3],[4],[3,4]]
	elif file in ['kddcup_10_percent.data','covtype.data']:
		modeState = [[2,3],[2,4],[4],[2]]
	else:
		modeState = [[3],[4],[3,4]]
		labeled = False

def run(path):
	f = open('run.log','a+')
	tmp = sys.stdout
	sys.stdout = f
	dpath = r'C:\Users\LiuLiang\Desktop\HCbirch\data\exp'
	#files = ['covtype.data']#,'kddcup_10_percent.data']#,'usps.data','covtype.data',]
	for file in files:
		print '++++++++++++++++++++++++++++++++++++++++'
		print '++++++++++++++++++++++++++++++++++++++++'
		print file
		setRunMode(file)
		filename = dpath + '\\' + file
		setPara.setValues(file)
		treeori,timeori = clustering.run_brich_ori(filename,'_ori'+'.pickle',path + file)
		for st in modeState:
			setPara.setMode(st)
			print '--------------'
			print str(st)
			treehr,timehr = clustering.run_hrbrich(filename,'_hr'+'.pickle',path+file+str(st))
			#treeori,timeori = run_brich_ori(filename,'ori_'+file+'.pickle',path)
			#ev.printinfor(treeori,birch_ori.Entry.d2)
			#ev.printinfor(treehr,hrbirch.Entry.d2)

	f.close()
	sys.stdout = tmp

def eva(path,labeled):
	f = open(r'eva.log','a+')
	tmp = sys.stdout
	sys.stdout = f
	#files = ['usps.data','covtype.data','kddcup_10_percent.data']
	#files = ['covtype.data']#,'kddcup_10_percent.data']#,'usps.data','covtype.data',]
	#files = ['covtype.data']#'kddcup_10_percent.data','usps.data','covtype.data',]
	for file in files:
		setPara.setValues(file)
		setRunMode(file)
		with open(path + file + '[4]groundLabels.pickle','rb') as ft:
			gnd = pickle.load(ft)
		with open(path +file+ '[4]normalizor.pickle','rb') as ft:
			norm = pickle.load(ft)
		evalute.setNormAndGnd(norm,gnd)
		print '++++++++++++++++++++++++++++++++++++++++'
		print '++++++++++++++++++++++++++++++++++++++++'
		print file
		with open(path + file + '_ori'+ '.pickle' ,'rb') as ft:
			ori_tree = pickle.load(ft)
			evalute.printinfor(ori_tree,birch_ori.Entry.d4,labeled)
		for st in modeState:
			setPara.setMode(st)
			print '--------------'
			print str(st)
			with open(path +file + str(st) + '_hr' + '.pickle' ,'rb') as ft:
				gv.cftree = pickle.load(ft)
				evalute.printinfor(gv.cftree,hrbirch.Entry.d4,labeled)
			with open(path +file + str(st) + '_hr' + '.pickle' ,'rb') as ft:
				gv.cftree = pickle.load(ft)
				evalute.printinfo_single(gv.cftree,hrbirch.Entry.d4,labeled)

	f.close()
	sys.stdout = tmp


if __name__ == '__main__':
	count = 20
	res_path = r'C:\Users\LiuLiang\Desktop\HCbirch\new_res\\' + str(count) + '\\'
	try:
		os.mkdir(res_path)
	except:
		pass
	#os.chdir(res_path)
	with open('commit.log','w') as f:
		comm = 'exp round,fruit is open and d4'
		f.write(comm)
	run(res_path)
	eva(res_path,labeled)
