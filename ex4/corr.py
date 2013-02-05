############### Complex NX Session 4 *Uli*Gui*Dani*Flo* ###############

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def Q(A, group, input, parts):
    """ Evaluate q in dependance of the group correlation in
        the matrix.
    """
    q = 0 
    for i in range(parts):
        for j in range(parts):
            if i!=j :
                delta = group[i] == group[j]
                q = q - (A[i][j] - 
                         np.sum(A,axis=0)[i] * np.sum(A,axis=0)[j]
                         /np.sum(A)) * delta
    return q

def switch_groop(group):
    for i in range(2):
        z = np.random.randint(parts)
        new_group = group.copy()
        if group[z] == 0:
            new_group[z] = 1
        elif group[z] == 1:
            new_group[z] = 0      
    return new_group
    
def matrix(input, parts):
    matrix = np.zeros(shape = (parts, parts))
    for i in range(len(input)):
        matrix[input[i][0]-1][input[i][1]-1] = 1
        matrix[input[i][1]-1][input[i][0]-1] = 1
    return matrix


# ========== MAIN ============

# --- init ---
input = np.loadtxt('karate.txt')
parts = int(np.max(input))
group = np.random.randint(2, size=parts )
A = matrix(input, parts)
eps = 1.002

Beta = 1 + 0.1*np.random.random()
counter = 0
Q_old = Q(A, group, input, parts)

# --- update ---
while Beta < 1E5:
    counter += 1
    new_group = switch_groop(group) 
    Q_new = Q(A,new_group,input, parts)
    z = np.random.random()

    if z < 1/( 1 + np.exp(-Beta * (Q_old-Q_new)) ):
        group = new_group.copy()
        Q_old = Q_new.copy()
    Beta = Beta**eps * eps**Beta        # chosen temperature funktion

    print 'Relaxing Q: ',  Q_old, '    ', ' Temperature: ', 1/Beta

# --- output ---
print 
print 'Extremal Q: ', Q_old 
print 'Iterations: ', counter
