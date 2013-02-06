############### Complex NX Session 4 *Uli*Gui*Dani*Flo* ###############

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
    
def matrix(input, parts):
    """ Create a matrix with dimensions parts*parts and represent
        correlations of 'input' as 0 and 1.
    """
    matrix = np.zeros(shape = (parts, parts))
    for i in range(len(input)):
        matrix[input[i][0]-1][input[i][1]-1] = 1
        matrix[input[i][1]-1][input[i][0]-1] = 1
    return matrix

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

#
#def matrix_sort(A, group, dim=2):
    #A = A.tolist()
    #Z = []
    #for g in range(len(group)):
        #if bool(group[g]) == True:
            #Z.append(A.pop(g-len(Z)))
            #print 'A',A
            #print 'Z',Z
            #print 'group', group
            #print 'g',g
            #print 'bool=', bool(group[g])
    #Z.append(A[0])
    #T = []
    #for j in range(len(Z)):
        #T.append([Z[i][:][j] for i in range(len(Z))])
    #Y = []    
    #for g in range(len(group)):
        #if bool(group[g]) == True:
            #Y.append(T.pop(g-len(Y)))
            #print 'T',T
            #print 'Y',Y
            #print 'group', group
            #print 'g',g
            #print 'bool=', bool(group[g])
    #Y.append(T[0])
    #return Y



# ========== MAIN ============

# --- init ---
input = np.loadtxt('karate.txt')
parts = int(np.max(input))                  # highest node value
group = np.random.randint(2, size=parts )   # random group assignment
A = matrix(input, parts)
eps = 1.102                                 # changerate of beta

Beta = 1 + 0.1*np.random.random()
counter = 0
Q_old = Q(A, group, input, parts)

# --- update ---
#while Beta < 1E5:
#    counter += 1
#    new_group = switch_groop(group) 
#    Q_new = Q(A,new_group,input, parts)
#    z = np.random.random()
#
#    if z < 1/( 1 + np.exp(-Beta * (Q_old-Q_new)) ):
#        group = new_group
#        Q_old = Q_new
#    Beta = Beta**eps * eps**Beta        # chosen temperature function
#
## --- output ---
#    print 'Relaxing Q: ',  Q_old, '    ', ' Temperature: ', Beta
#print 
#print 'Extremal Q: ', Q_old 
#print 'Iterations: ', counter









######################################################################







    #vtemp = list(A) #np.vsplit(A, len(A))
    #Z = []   #[0 for l in range(len(group))] 
    #counter = 0
    #for d in range(dim):
        #for g in range(len(group)):
            #if group[g] == d:
                #Z.append(A[g])
                #print Z
    #A = Z
    #print 'LOOOL'
    #Y = []
    #for d in range(dim):
        #for g in range(len(group)):
            #if group[g] == d:
                #Y.append(A[ for i in len(A)][g])
                #print Y
     #htemp = np.hstack(Z)
    #print np.vstack(Y)
    #counter = 0
    #for d in range(dim):
        #for g in range(len(group)):
            #if group[g] == d:
                #Z[:][counter]=htemp[:][g]
                #print 'ycounter', Z[:][counter]
                #print 'ycounter', Z[counter]
                #counter += 1
    #return np.vstack(htemp)
    #htemp = np.vsplit(Z, len(Z))
    #Z = np.array([])
    #for d in range(dim):
        #for g in range(len(group)):
            #if group[g] == d:
                #Z = np.hstack((Z, htemp[g]))
            #print 'group_g', htemp[g]
            #print 'colz', Z
    #return Z

#def matrix_sort(A, group, dim=2):
    #rowZ = []
    #for d in range(dim):
        #for g in range(len(group)):
            #if group[g] == d:
                #rowZ = np.append(rowZ, A[g])
            #print 'group_g', group[g]
            #print 'rowz', rowZ
    #colZ = []
    #for d in range(dim):
        #for g in range(len(group)):
            #if group[g] == d:
                #colZ = np.append(colZ, rowZ[:][g])
            #print 'group_g', group[g]
            #print 'colz', colZ
    #return colZ


