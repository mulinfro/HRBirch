import numpy as np
import gb_variables as gv
import birch_ori
import hrbirch
import time,random,pickle
import evalute,setPara,clustering
def run():
	import sys
	f = open('run.log','a+')
	tmp = sys.stdout
	sys.stdout = f
	path = r'C:\Users\LiuLiang\Desktop\HCbirch\data\exp'
	res = r'C:\Users\LiuLiang\Desktop\HCbirch\result\\'
	files = ['usps.data','covtype.data','kddcup_10_percent.data']
	modeState = [[]]#[[2],[3],[4],[2,3,4]]
	for file in files:
		print '++++++++++++++++++++++++++++++++++++++++'
		print '++++++++++++++++++++++++++++++++++++++++'
		print file
		setPara.setValues(file)
		for st in modeState:
			setPara.setMode(st)
			print '--------------'
			print str(st)
			filename = path + '\\' + file
			treehr,timehr = clustering.run_hrbrich(filename,'_hr'+'.pickle',res+file+str(st))
			print 'ori====================================='
			treeori,timeori = clustering.run_brich_ori(filename,'ori_'+'.pickle',res+file)
			#ev.printinfor(treeori,birch_ori.Entry.d2)
			#ev.printinfor(treehr,hrbirch.Entry.d2)

	f.close()
	sys.stdout = tmp

def eva():
	import sys
	f = open('eva.log','a+')
	tmp = sys.stdout
	sys.stdout = f
	path = r'C:\Users\LiuLiang\Desktop\HCbirch\data\exp'
	res = r'C:\Users\LiuLiang\Desktop\HCbirch\result\\'
	files = ['usps.data','covtype.data','kddcup_10_percent.data']
	modeState = []#[[2],[3],[4],[2,3,4]]
	for file in files:
		print '++++++++++++++++++++++++++++++++++++++++'
		print '++++++++++++++++++++++++++++++++++++++++'
		print file
		setPara.setValues(file)
		for st in modeState:
			setPara.setMode(st)
			print '--------------'
			print str(st)
			filename = path + '\\' + file
			treehr,timehr = clustering.run_hrbrich(filename,'_hr'+'.pickle',res+file+str(st))
			#treeori,timeori = run_brich_ori(filename,'ori_'+file+'.pickle',res)
			#ev.printinfor(treeori,birch_ori.Entry.d2)
			#ev.printinfor(treehr,hrbirch.Entry.d2)

if __name__ == '__main__':
	run()
