############################ Ulrich & Daniel, Guilherme & Florian (2013) #######

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pylab import *

def int2bin(size, n):
    """ Convert an integer n to binary number. Return a list with length 'size'. 
        Each element of the list contains one binary digit. The lowest power 
        of 2 corresponds to the first element.
        Expect two integers, return list.
    """
    assert type(n) == int
    assert type(size) == int
    v = [0] * size                      # initialze list with length 'size'
    for i in range(len(bin(n)) - 2):    # -2 removes 0bxxxxxx
        v[i] = int(bin(n)[ -i - 1])     # convert string to int and reverse order
    return v    

def bin2int(v):
    """ Convert a binary list v into an integer n.
    """
    assert type(v) == list
    n = 0
    for i in range(len(v)):             # add all powers of 2
        assert v[i] == 1 or v[i] == 0
        n = n + 2**i * v[i]             # where i_th element of the vector is 1
    return n

def readVal(valType, reqMsg='Input a value: ', errMsg='Wrong type, try again.'):
    """ Return 'value' if it is of the requested type 'valType',
        else try again. Optional: reqMsg and errMsg.
    """
    while True:
        value = raw_input(reqMsg)
        try:
            value = valType(value)
            return value
        except:
            print(errMsg)


#obsolete====
#def generate_evolution_table(N, k):
    #""" N x 2**k array of random outputs
        #first index: node
        #second index: output; index encodes binary input configuration
    #""" 
    #t = np.random.random_integers(0,1, size=(N, 2**k))
    #return t

def biased_bool(N, k, p):
    """ Return N x 2**k array of random boolean with bias p.
    """
    return (np.random.random((N, 2**k)) < p)*1

# get output of node i with input configuration e_v from table above
def evolution_table(Table, i,v):
    t = Table[i][bin2int(v)]
    return t

def return_sequence(N, k, graph, curr_state, Table):
    v_return=[None]*N
    for node in range(N):
        neighbor = graph.neighbors(node)
        neighbor_state = np.array(())
        for j in range(k):
            neighbor_state = np.append(neighbor_state, 
                                       curr_state[neighbor[j]]
                                      )
        v_return[node] = evolution_table(Table, node, neighbor_state)
    return v_return

# -------- main --------- 

#N = int(raw_input('Number of nodes: '))
#k = int(raw_input('Number of inputs: '))
#p = float(raw_input('Probability bias: '))

N = readVal(int,'Number of nodes: ', 'Input must be a positive Integer.')
k = readVal(int,'Number of inputs: ', 'Input must be a positive Integer.')
p = readVal(float,'Probability bias: ', 'Input must be a float between 0.0 and 1.0.')

np.random.seed()

G = nx.random_regular_graph(k, N)

#bool_operation = generate_evolution_table(N, k)
bool_operation = biased_bool(N, k, p)

state_space = nx.DiGraph()

for i in range(2**N):                   # int(i), int(bin2int) --> i, bin2int; matters?
    state_space.add_edge(i, bin2int(N,
                return_sequence(N, k, G, int2bin(N,i),
                        bool_operation)
               )
           )

# plot alternative, pydot in comment area:
pos = nx.graphviz_layout(state_space,prog="twopi",root=0)
nx.draw(state_space,pos,with_labels=False,
   alpha=0.4,node_size=10)

plt.title('N nodes with k connections')
plt.savefig("output.pdf")



####################################################################################################
####################################################################################################
####################################################################################################
# below removed stuff from this ex

#import copy
#import random as rd
#import pydot


#def evolution_table_k3(Table,i,E1,E2,E3):
#   return Table[i][4*E1+2*E2+E3] 
#   
#def evolution_table_k2(Table,i,E1,E2):
#   return Table[i][2*E1+E2] 
    
# table=generate_evolution_table(N,k)

#for i in range(N):
#    state_space.add_node(i)

#state_space_conn=np.array(())
#for i in range(2**N):
#    state_space_conn=np.append(state_space_conn, 
#                               np.random.randint(0, 2**N))


#def plot_graph(graph):
    #nx.draw(graph)
    #show()


#graph = nx.to_pydot(state_space,pos)
#graph.write_pdf('output.pdf')

#plot_graph(state_space)



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
# below ex2 as reference










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
## THIS LINE COULD BE DELETING YOUR WHOLE DRIVE... BEWARE =))))
#ax1.set_title('$\gamma$ = '+ str(gamma))
#
#ax2 = fig.add_subplot(111)
#ax2.plot(x2, y2, color='blue', linewidth=2, label="Kill largest hubs")
#
#plt.legend(loc='upper right')
#plt.show()
