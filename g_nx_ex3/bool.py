############################ Ulrich & Daniel, Guilherme & Florian (2013) ######

import numpy as np
import networkx as nx
import random as rd
import matplotlib.pyplot as plt
import copy
from pylab import *
import pydot

N = 10     #number of nodes
k = 2           # number of connections

# N x 2**k dimensional array of random outputs
# first index: node
# second index: output; index encodes binary input configuration
def generate_evolution_table():
    return np.random.random_integers(0,1, size=(N,2**k))

# get output of node i with input configuration e_v from table above
def evolution_table(Table,i,e_v):
    return Table[i][bin_to_int(k,e_v)]

#def evolution_table_k3(Table,i,E1,E2,E3):
#   return Table[i][4*E1+2*E2+E3] 
#   
#def evolution_table_k2(Table,i,E1,E2):
#   return Table[i][2*E1+E2] 
    
# table=generate_evolution_table(N,k)

#transform integer to binary-array
def int_to_bin(size,n):
    # v=np.array([])
    # tmp=n%2**(i+1)
    # v=np.append(v,tmp)
    # n-=tmp*2**i   
    v=[None]*size
    for i in range(0,size):
        if n>=(2**(size-i-1)):
            v[size-i-1]=1
            n=n-2**(size-i-1)
        else:
            v[size-i-1]=0   
    return v
    
#transform binary-array to integer
def bin_to_int(size,v):
    n=0
    for i in range(0,size):
        n=n+2**i*v[i]
    return n    

    
def return_sequence(graph,curr_state,Table):
    v_return=[None]*N
    for node in range(N):
        neighbor = graph.neighbors(node)
        neighbor_state=np.array(())
        for j in range(k):
            neighbor_state=np.append(neighbor_state,curr_state[neighbor[j]])
        v_return[node] = evolution_table(Table,node,neighbor_state)
    return v_return

def plot_graph(graph):
    nx.draw(graph)
    show()

G = nx.random_regular_graph(k,N)

bool_operation=generate_evolution_table()

state_space = nx.DiGraph()
    
#for i in range(N):
#    state_space.add_node(i)

#state_space_conn=np.array(())
#for i in range(2**N):
#    state_space_conn=np.append(state_space_conn, 
#                               np.random.randint(0, 2**N))

for i in range(2**N):
    state_space.add_edge(int(i),int(bin_to_int(N,return_sequence(G,int_to_bin(N,i),
                                                        bool_operation))))

graph = nx.to_pydot(state_space)
graph.write_png('test1.png')

#plot_graph(state_space)
















#def generate_graph(gamma, kmin, kmax, dk):
#    # map probability to power law:
#    p = np.random.random(graph_size)
#    C = (-gamma+1) / ((kmax+dk)**(-gamma+1) - (kmin+dk)**(-gamma+1))
#    k = ((-gamma+1)/C * p + (kmin+dk)**(-gamma + 1)) ** (1/(-gamma+1)) - dk
#    sequence = map(int, sorted(np.round(k)))                    # map sorted intervalls to int
#    if np.mod(np.sum(sequence), 2) != 0:                        # if odd, get rid of last element
#        sequence[-1]-=1                                 
#    G = nx.configuration_model(sequence)                
#    G = nx.Graph(G)                                             # remove multiple connections
#    return G
#
#def kill_random_node(at_once):
#    kill_list = rd.sample(range(0, G.number_of_nodes() - 1),    # generate sample of removal 
#                          at_once)                              
#    removable_nodes = G.nodes()                                 # which nodes are available
#    for i in range(len(kill_list)):                             # remove all elements specified
#        G.remove_node(removable_nodes[kill_list[i]])            # by kill_list
#
#def kill_hub(at_once):
#    for i in range(at_once):
#        max_index=np.argmax(edges_per_node)                     # returns index of largest node
#        edges_per_node[max_index]=0                             # set node as smallest in list
#        G.remove_node(max_index)                                # remove node from graph
# 
#def plot_graph():
#    nx.draw(G)
#    show()
#    
## ------ main ------  
#graph_size = 100000                                             # initialize variables
#gamma = 3.5
#kmin = 1.0
#kmax = float(graph_size-1)
#dk = 0.0
#killstep = 50                                                   # how many are removed at once
#y1 = np.array([])
#x1 = np.array([])
#y2 = np.array([])
#x2 = np.array([])
#
#G0 = generate_graph(gamma,kmin,kmax,dk)                         # generates graph
#Gcc0 = len(nx.connected_components(G0)[0])                      # size of giant component
#
#
#Gcc = copy.copy(Gcc0)                                           # make instance for Random kill
#G = G0.copy()
#killed = 0.0
#
#while Gcc/Gcc0 > 0.001 and G.number_of_nodes > killstep:
#    kill_random_node(killstep)
#    killed += killstep
#    print 'Random Kill: ', killed/graph_size                    
#    Gcc = float(len(nx.connected_components(G)[0]))
#    x1 = np.append(x1, killed/graph_size)
#    y1 = np.append(y1, Gcc/Gcc0)
#
#
#Gcc = copy.copy(Gcc0)                                           # make instance for Hub kill
#G = G0.copy()
#killed = 0.0
#
#edges_per_node = np.array(())                                   # count edges for each node
#for i in range(graph_size):
#    edges_per_node = np.append(edges_per_node, int(len(G.edges(i))))
#
#while Gcc/Gcc0 > 0.001 and G.number_of_nodes > killstep:
#    kill_hub(killstep)
#    killed += killstep
#    print 'Hub Kill: ', killed/graph_size
#    Gcc = float(len(nx.connected_components(G)[0]))
#    x2 = np.append(x2, killed/graph_size)
#    y2 = np.append(y2, Gcc/Gcc0)
#
#
#fig = plt.figure()
#
#ax1 = fig.add_subplot(111)
#ax1.plot(x1, y1, color='red', linewidth=2, label="Kill random nodes")
#ax1.set_title('$\gamma$ = '+ str(gamma))
#
#ax2 = fig.add_subplot(111)
#ax2.plot(x2, y2, color='blue', linewidth=2, label="Kill largest hubs")
#
#plt.legend(loc='upper right')
#plt.show()
