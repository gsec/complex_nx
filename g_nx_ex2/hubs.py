########################### network excercise 2 ###############################
#########################--- killing a network ---#############################
######################### Guilherme Stein (2013) ##############################
######### in collaboration with: Florian, Ulrich & Daniel #####################



import numpy as np
from pylab import *
from scipy import optimize  
from scipy import stats
import networkx as nx
import random as rd
import matplotlib.pyplot as plt

def generate_graph(gamma,kmin,kmax,dk):
	rnd = np.random.random(graph_size)
	C = (-gamma+1) / ((kmax+dk)**(-gamma+1) - (kmin+dk)**(-gamma+1))
	k = (((-gamma+1)/C)*rnd+(kmin+dk)**(-gamma+1))**(1/(-gamma+1))-dk
	sequence=map(int,sorted(np.round(k)))
	if np.mod(np.sum(sequence),2)!=0:
		sequence[-1]-=1
	#print(sequence,np.sum(sequence))
	G = nx.configuration_model(sequence,Graph)
    # G = nx.random_degree_sequence_graph(sequence)
	G.remove_edges_from(G.selfloop_edges())
	return G

def plot_graph():
	nx.draw(G)
	show()
	return

def random_kill_a_node(number):
	kill_list=rd.sample(range(0,G.number_of_nodes()-1),number)
	# kill_list=np.random.random_integers(0,G.number_of_nodes()-1,number)
	rem_nodes=G.nodes()	
	tmp=np.array([])
	for i in range(len(kill_list)):
		tmp=np.append(tmp,rem_nodes[kill_list[i]])
	G.remove_nodes_from(tmp)

# def hub_kill_a_node(number):
	# kill_list=G.number_of_edges()[:number] #rd.sample(range(0,G.number_of_nodes()-1),number)
	# rem_nodes=G.nodes()	
	# tmp=np.array([])
	# for i in range(len(kill_list)):
		# tmp=np.append(tmp,rem_nodes[kill_list[i]])
	# G.remove_nodes_from(tmp)
	
# --- main ---	
graph_size=10000
gamma=2.5
kmin=1.0
kmax=float(graph_size-1)
dk=0.0
killed=0.0
killstep=50
y=np.array([])
x=np.array([])

G=generate_graph(gamma,kmin,kmax,dk)
Gcc0=len(nx.connected_components(G)[0])
Gcc=Gcc0


while Gcc/Gcc0 > 0.01:
	random_kill_a_node(killstep)
	killed+=killstep
	Gcc=len(nx.connected_components(G)[0])*1.0
	x=np.append(x,killed/graph_size)
	y=np.append(y,Gcc/Gcc0)
	print(killed/graph_size)
# print(x)	
# print(y)


fig=plt.figure()
legend(loc=2)
ax=fig.add_subplot(111)
ax.plot(x,y,color='red',label='o')
plt.show()





#plot_graph()





# N=100000
# Punkte=200
# kv=np.linspace(0,4,Punkte)
# S=np.linspace(0,0,Punkte)
# s=np.linspace(0,0,Punkte)

# i=0
# for k in kv:
	# p=k/N
	# G=nx.fast_gnp_random_graph(N,p)
	# Gcc=nx.connected_components(G)
	# S[i]=len(Gcc[0])*1.0/N
	# s[i]=(N-len(Gcc[0]))*1.0/(len(Gcc)-1)	
	# i=i+1
	# print(i)

# sp=1/(1-kv+kv*S)
# logS=log(S)
# logk=log(kv-1)

# # --- fit	
# fitfunc = lambda p, x: p[0]+p[1]*x	
# def residuals(p, y, x):
	# err = y-fitfunc(p,x)
	# return err
# p0 = [1,1]	
# p1, success = optimize.leastsq(residuals, p0[:], args=(logS, logk))
# print(logS,logk)
	
# # --- plot	
# fig=plt.figure()
# legend(loc=2)
# ax=fig.add_subplot(111)
# ax.set_ylabel(u'S')
# ax.set_xlabel(u'<k>')
# ax.plot(kv,S,color='blue',label='S')
# show()

# fig=plt.figure()
# legend(loc=2)
# ax=fig.add_subplot(111)
# ax.set_ylabel(u'S')
# ax.set_xlabel(u'<k>')
# ax.plot(kv,s,color='red',label='s')
# show()

# fig=plt.figure()
# legend(loc=2)
# ax=fig.add_subplot(111)
# ax.set_ylabel(u'log(S)')
# ax.set_xlabel(u'log(<k>-1)')
# ax.plot(logk,fitfunc(p1,logk),color='green',label='fitfunc')
# ax.plot(logk,logS,color='blue',label='logS')
# print(p1)
# show()	
