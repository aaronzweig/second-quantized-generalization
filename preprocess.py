import numpy as np

def vertex_index(i, j, n):
    r = n * (n+1) / 2 - (n-i) * (n-i+1) / 2
    return int(r + j - i)

# TEST:
# M = 5
# for i in range(M+1):
#     for j in range(i, M+1):
#         print(i,j, vertex_index(i,j,M+1))



#M: total number of orbitals
#N: total number of particles

#A is potential matrix: M x M
#U is coulumb 4-tensor: M x M x M x M
#X is additional orbital feature matrix: M x F
#Y is additional pairwise orbital feature matrix: M x M x F

#TODO: include features to indicate single-orbital vs double-orbital vertices
#TODO: treat edges as features rather than lengths to distinguish the connections (i, _) -- (i, j)

def build_graph(A, U, X, Y):
    M = X.shape[0]
    F = X.shape[1]
    
    V = int((M+1)*(M+2)/2)
    W = np.zeros((V, V))
    Z = np.zeros((V, F))

    
    #(i, j) indicates a vertex for orbital pair i-j
    #(i, M) indicates a vertex for the single orbital i
    for i in range(M+1):
        for j in range(i, M+1):
            u = vertex_index(i,j,M+1)
            if j < M:
                Z[u] = Y[i][j]
            elif i < M:
                Z[u] = X[i]
            else:
                Z[u] = np.zeros(F)
            
            for k in range(M+1):
                for l in range(k, M+1):
                    v = vertex_index(k,l,M+1)

                    if j < M and l < M:
                        W[u,v] = U[i,j,k,l]
                    elif i < M and j == M and k < M and l == M:
                        W[u,v] = A[i,k]
                    elif j == M and l < M and (i == k or i == l):
                        W[u,v] = 1
                    elif l == M and j < M and (k == i or k == j):
                        W[u,v] = 1
                    
                        
    return W, Z
