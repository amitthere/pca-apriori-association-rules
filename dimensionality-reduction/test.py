import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

test = np.loadtxt('../data/pca_a.txt', delimiter="\t", dtype='str').shape[1]
#print(test)

initial = np.loadtxt('../data/pca_a.txt', delimiter = "\t", dtype='str')
#print(initial)
lastcol = initial[:,-1]
#print(lastcol)
matrix = np.loadtxt('../data/pca_a.txt', delimiter = "\t", usecols = range(test-1))
#print(matrix.shape)
row = len(matrix)
#print(row)
t = np.asmatrix(lastcol, dtype='U')
#print(t.shape)
t = np.transpose(t)
#print(t.shape)
matrix_t = np.transpose(matrix)
#print(matrix_t.shape)
mean = np.mean(matrix, axis=0)
#print(mean)
normalize = matrix - mean
#print(normalize)
col = normalize.shape[1];
#print(col)
matrix_calc = np.matrix(normalize[0,:])
#print(matrix_calc)
for i in range(1,col):
    mat_col = np.matrix(normalize[i,:])
    matrix_calc = np.vstack((matrix_calc,mat_col))
#print(matrix_calc)
covmat = np.cov(matrix_calc)
#print(covmat)
eigvals, eigvecs = np.linalg.eig(covmat)
#print("eigvals", eigvals)
#print("____________")
#print("eigvecs", eigvecs)
#print("____________")
idx = eigvals.argsort()[::-1]
eigvecs = eigvecs[idx]
eigvecs = eigvecs[:,idx]
w = eigvecs[:,:2]
#print('w', w)
#print("w", w)
#print("____________")
w_trans = np.transpose(w)
#rint(w_trans)
matrix_transpose = np.transpose(matrix)

final_matrix = np.dot(w_trans, matrix_transpose)
#final_matrix = np.transpose(ans)

#print("ans", final_matrix.shape)
#print(t)

temp = np.concatenate((final_matrix, t), axis=1)
print(temp)



svd = TruncatedSVD(n_components=2)
svd_result = svd.fit_transform(matrix)
svd_result = np.concatenate((svd_result, t), axis = 1)
#print(svd_result)

tsne = TSNE(n_components = 2)
tsne_result = tsne.fit_transform(matrix)
tsne_result = np.concatenate((tsne_result, t), axis = 1)
#print(tsne_result)