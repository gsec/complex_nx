import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pylab import *
import operator as op
import sys, select

def Q(Matrix,group,karate):
	q=0	
	for i in range(int(np.max(karate))):
		for j in range(int(np.max(karate))):
			if (i!=j):
				delta = (group[i]==group[j])
				q=q-(Matrix[i][j]-np.sum(Matrix,axis=0)[i]*np.sum(Matrix,axis=0)[j]/np.sum(Matrix))*delta
	return q

def generate_graph(karate):
	"""erzeugt einen Graphen aus Wertepaaren als Kanten"""
	G = nx.Graph()
	G.add_edges_from(karate)
	return G

def generate_groups(karate):
	group = np.zeros(np.max(karate))
	group = np.random.randint(2,size=(int(np.max(karate))))
	return group
	
def generate_new_group(group):
	for i in range(2):
		z= np.random.randint((np.max(karate)))
		new_group=group.copy()
		if group[z]==0:
			new_group[z]=1
		elif group[z]==1:
			new_group[z]=0		
	return new_group
	
def Adj(karate):
	Adj=np.zeros(shape=(int(np.max(karate)),int(np.max(karate))))
	for i in range(len(karate)):
		Adj[karate[i][0]-1][karate[i][1]-1]=1
		Adj[karate[i][1]-1][karate[i][0]-1]=1
	return Adj

karate = np.loadtxt('karate.txt')
G = generate_graph(karate)
group = generate_groups(karate)
Matrix=Adj(karate)

# --- update --- 
betta = 0.01
Q_alt=Q(Matrix,group,karate)
for i in range(1000):
	new_group=generate_new_group(group)	
	Q_neu=Q(Matrix,new_group,karate)
	z = np.random.random()
	# print "z", z
	# print "Q_alt", Q_alt
	# print "Q_neu", Q_neu
	# print "prob", 1/(1+np.exp(-betta*(Q_alt-Q_neu)))
	# print "exp", np.exp(-betta*(Q_alt-Q_neu))
	# print ""
	if z < 1/(1+np.exp(-betta*(Q_alt-Q_neu))):
		group=new_group.copy()
		Q_alt=Q_neu.copy()
	betta = betta*1.01
	print("qneu", Q_alt, "  betta: ", betta)

print Q(Matrix,group,karate)

	


# print(np.zeros(np.max(karate)))