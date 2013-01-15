# ######################### graph excercise 1 #################################
# ####################### Guilherme Stein (2013) ##############################

import networkx as nx
import matplotlib.pyplot as plt

def gc(N,p):                                # generate cluster for N nodes
    h = nx.fast_gnp_random_graph(N, p)      # with probability p
    cluster = nx.connected_components(h)
    lc = len(cluster[0])                    # size of largest cluster
    k = N * p                               # average connectivity as on p.14
    mc = 0.                                 # initialize mean cluster
    if len(cluster) > 1:
      for i in range(len(cluster)-1):
        mc += len(cluster[i+1])             # accumulate cluster sizes
      mc = mc*1./(len(cluster)-1)           # and normalize
    return lc, mc

def gcvary(nodes, step):
    listP = []
    listL = []
    listS = []
    i = 0
    span = 100
    while i < span:
        i = i + step                        # until 5 and
        p = i * 5 / (nodes*span)            # normalize
        lc, s = gc(nodes,p)
        listP.append(p)
        listL.append(lc)
        listS.append(s)
        print i                            # show progress
    return listP, listL, listS

# ----------------------------------------
nodes, step = 100000, 0.2
P, L, S = gcvary(nodes, step)

varK = [x*nodes for x in P]                 # probability to mean degree
normL = [x/float(nodes) for x in L]         # normalize giant component

plt.figure(figsize = (15, 10), dpi = 80)
plt.plot(varK, S, color='blue', linewidth=2, label="mean cluster")
plt.plot(varK, normL, color='red', linewidth=2, label="giant component")
plt.legend(loc='upper right')

plt.show()
