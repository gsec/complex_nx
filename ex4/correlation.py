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

# replaced in main
#def generate_groups(input,dim):
    #group = np.zeros(parts)
    #group = np.random.randint(dim,size=parts)
    #return group
    
def generate_new_group(group, dim, num=1):
    """ Manipulate a vector by changing entrys into other values.
    """
    instance = group.copy()
    for i in range(num):
        z = np.random.randint(parts)
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

# ---- could be replaced by fct 'matrix'
def Adj(input):
    Adj=np.zeros(shape=(parts,parts))
    for i in range(len(input)):
        Adj[input[i][0]-1][input[i][1]-1]=1
        Adj[input[i][1]-1][input[i][0]-1]=1
    return Adj
# -----------

def swap_blanck(input):
    """ Initialize unity matrix of dimension (parts x parts)
    """ 
    swap=np.zeros(shape=(parts,parts))
    for i in range(parts):
        swap[i][i]=1    
    return swap


def swap_order(group,input,dim):
    """ Create a transformation matrix.
    """ 
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
    """ Coordinate transformation of A through S^T * A * S
    """ 
    adj_matrix = np.mat(Adj)
    Adj_order = Swap.transpose() * adj_matrix * Swap 
    Adj_order=np.asarray(Adj_order)
    return Adj_order

def plot(Adj_ordered):
    plot_matrix=np.zeros((parts,parts))
    for i in range(parts):
        for j in range(parts):
            plot_matrix[i][j]=Adj_ordered[i][parts-j-1]
    pcolor(plot_matrix)
    show()
    return      # what is this?


############ MAIN ############


# --- init ---
# number of groups; rather too many then too few
dim = 5
input = np.loadtxt('karate.txt')
# parts specifies how many elements there are in general
# here it is the highest node value
parts = int(np.max(input))                 
counter = 0                             
kB = 1E-19           # scaling factor

group = np.random.randint(dim, size=parts )   # inital random group assignment
#group = generate_groups(input,dim)
Matrix=Adj(input)

# Specification of beta, set 'eps' to 1.002 for faster change
# but less likely to be in equillibrium
noise = 0.01*np.random.random()
eps = 1.002                            
Beta = 1.001 + noise

Q_old = Q(Matrix, group, input, parts)


# --- update --- 
while Q_old > -73 and Beta < 1E250:          #tough criterias...Beta < 1E80:
    counter += 1
    new_group = generate_new_group(group,dim)
    Q_new = Q(Matrix,new_group,input,parts)
    z = np.random.random()
    if z < 1/ (1 + np.exp( -Beta * (Q_old-Q_new) )):
        group = new_group.copy()
        Q_old = Q_new.copy()
    Beta = Beta ** eps#**Beta        # arbitrary temperature function
    print"Q: ", Q_old, " Temperature: ", 1/(Beta*kB)

ordered_Matrix = Adj_order(swap_order(group,input,dim),Matrix)
original_matrix = np.asarray(Adj(input))



# ---- output -----
print 
print 'Extremal Q: ', Q_old
print 'Iterations: ', counter

#plot(original_matrix)
plot(ordered_Matrix)
