############### Complex NX Session 4 *Uli*Gui*Dani*Flo* ###############

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pylab import *
    
def Q(A, group, input, parts):
    """ Evaluate q in dependance of the group correlation in the matrix.
    """
    q = 0 
    for i in range(parts):

        for j in range(parts):
            if i!=j :
                Delta = group[i] == group[j]
                q = q - (A[i][j] - 
                         np.sum(A,axis=0)[i] * np.sum(A,axis=0)[j]
                         /np.sum(A)) * Delta
    return q

def switch_groop(group, num=1):
    """ Flips 'num'(default=1) random bits of a list. 
    """
    for i in range(num):
        new_group = group.copy()
        z = np.random.randint(len(group))
        if group[z] == 0:
            new_group[z] = 1
        elif group[z] == 1:
            new_group[z] = 0
    return new_group

#########################################################################
#import numpy as np
#import networkx as nx
#import matplotlib.pyplot as plt
#import operator as op
#import sys, select

#def Q(Matrix,group,karate):
    #q=0    
    #for i in range(int(np.max(karate))):
        #for j in range(int(np.max(karate))):
            #if (i!=j):
                #Delta = (group[i]==group[j])
                #q=q-(Matrix[i][j]-np.sum(Matrix,axis=0)[i]*np.sum(Matrix,axis=0)[j]/np.sum(Matrix))*Delta
    #return q


#def generate_graph(input):
    #"""erzeugt einen Graphen aus Wertepaaren als Kanten"""
    #G = nx.Graph()
    #G.add_edges_from(input)
    #return G

def generate_groups(input,dim):
    group = np.zeros(parts)
    group = np.random.randint(dim,size=parts)
    return group
    
def generate_new_group(group,dim):
    z = np.random.randint(parts)
    instance = group.copy()
    while instance[z] == group[z]:
        instance[z] = np.random.randint(dim)
    return instance
    
def matrix(input, parts):
    """ Create a matrix with dimensions parts*parts and represent
        correlations of 'input' as 0 and 1.
    """
    matrix = np.zeros(shape = (parts, parts))
    for i in range(len(input)):
        matrix[input[i][0]-1][input[i][1]-1] = 1
        matrix[input[i][1]-1][input[i][0]-1] = 1
    return matrix

def Adj(input):
    Adj=np.zeros(shape=(parts,parts))
    for i in range(len(input)):
        Adj[input[i][0]-1][input[i][1]-1]=1
        Adj[input[i][1]-1][input[i][0]-1]=1
    return Adj

def swap_blanck(input):
    swap=np.zeros(shape=(parts,parts))
    for i in range(parts):
        swap[i][i]=1    
    return swap


def swap_order(group,input,dim):
    print group
    Swap=np.mat(swap_blanck(input))
    for k in range(dim):
        SSwap=np.mat(swap_blanck(input))
        for l in range(dim-1):
            swap=swap_blanck(input)
            for i in range(parts):
                for j in range(parts-i-1): ###
                    n = parts-j-1      ###
                    if n > i and l < k:     
                        if group[i]==k and group[n]==l: 
                            group[i]=l
                            group[n]=k
                            swap[i][n]=1
                            swap[n][i]=1
                            swap[i][i]=0
                            swap[n][n]=0
#           np.savetxt('swap.txt', swap, fmt='%-2.0f')
            SSwap = SSwap*np.mat(swap)
#       np.savetxt('SSwap.txt', SSwap, fmt='%-2.0f')
        Swap = Swap*SSwap
#   swap = np.mat(swap[:,:,k])
    print group
    np.savetxt('Swap.txt', Swap, fmt='%-2.0f')
    return Swap

def Adj_order(Swap,Adj):
    adj_matrix = np.mat(Adj)
    Adj_order=Swap.transpose()*adj_matrix*Swap #transpose!
    Adj_order=np.asarray(Adj_order)
    return Adj_order

def plot(Adj_ordered):
    plot_matrix=np.zeros((parts,parts))
    for i in range(parts):
        for j in range(parts):
            plot_matrix[i][j]=Adj_ordered[i][parts-j-1]
    pcolor(plot_matrix)
    show()
    return

############ MAIN ############

# --- init ---
#input = np.loadtxt('karate.txt')
#group = np.random.randint(2, size=parts )   # random group assignment
#A = matrix(input, parts)
#eps = 1.102                                 # changerate of beta

#Beta = 1 + 0.1*np.random.random()
#counter = 0
#Q_old = Q(A, group, input, parts)


dim = 4
input = np.loadtxt('karate.txt')
parts = int(np.max(input))                  # highest node value
#G = generate_graph(input)
group = generate_groups(input,dim)
Matrix=Adj(input)
iteration = 1000
#eps = 0.02
eps = 1.002                                 # changerate of beta
counter = 0
Beta = 1.0 #+ 0.1*np.random.random()

# --- update --- 
#Beta = 0.01
Q_alt=Q(Matrix,group,input,parts)
#for i in range(iteration):
while Beta < 1E6:
    counter += 1
    new_group=generate_new_group(group,dim)
    print 'g,ng: ', group, new_group
    if (new_group != group).any():  
        Q_neu=Q(Matrix,new_group,input,parts)
        z = np.random.random()
        # print "z", z
        # print "Q_alt", Q_alt
        # print "Q_neu", Q_neu
        # print "prob", 1/(1+np.exp(-Beta*(Q_alt-Q_neu)))
        # print "exp", np.exp(-Beta*(Q_alt-Q_neu))
        # print ""
        if z < 1/(1+np.exp(-Beta*(Q_alt-Q_neu))):
            group=new_group.copy()
            Q_alt=Q_neu.copy()
        #Beta = Beta+eps
        Beta = Beta * eps**Beta        # arbitrary temperature function
        print("qneu", Q_alt, "  Beta: ", Beta)

print Q(Matrix,group,input,parts)

ordered_Matrix = Adj_order(swap_order(group,input,dim),Matrix)
original_matrix = np.asarray(Adj(input))
#plot(original_matrix)
plot(ordered_Matrix)


# print(np.zeros(parts))
print 'Iterations: ', counter
