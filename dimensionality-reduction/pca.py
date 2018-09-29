import numpy as np
from sklearn.decomposition import TruncatedSVD
import operator
from sklearn.manifold import TSNE

class Import:
    """
    Imports data from various sources.
    Currently, just tab-delimited files are supported
    """

    def __init__(self, file, ftype):
        self.data = None
        self.file = file
        if ftype == "TAB":
            self.import_tab_file(self.file)

    def import_tab_file(self, tabfile):
        self.data = np.genfromtxt(tabfile, dtype=str, delimiter='\t')

def matrix_manipulation(file):
    matrix = file.data[:,0:4]
    m1 = np.array(matrix.astype(np.float))
    last_col = file.data[:,-1]
    last_col_m = np.asmatrix(last_col, dtype='U')
    last = last_col_m.transpose()
    return m1, last

def eigen_values(file):
    m1, last = matrix_manipulation(file)
    m1_transp = m1.transpose()
    #print(m1.shape)

    mean_vector1 = np.mean(m1, axis= 0)
    #print(mean_vector1)

    centered1 = m1 - mean_vector1
    #print(centered1)

    covprep = centered1.T
    covar = np.cov(covprep)
    #print(covar)
    eig_val, eig_vec = np.linalg.eigh(covar)
    print(eig_val)
    print('____________')
    print(eig_vec)
    print('____________')

def svd_function(file):
    matrix, lastcol = matrix_manipulation(file)
    svd = TruncatedSVD(n_components=2)
    svd_res = svd.fit_transform(matrix)
    svd_res = np.concatenate((svd_res, lastcol), axis = 1)
    return svd_res
    print(svd_res)

def tsne_function(file):
    m1, lastcol = matrix_manipulation(file)
    tsne = TSNE(n_components=2)
    tsne_res = tsne.fit_transform(m1)
    tsne_result = np.concatenate((tsne_res, lastcol), axis= 1)
    print(tsne_result)
    return tsne_result

def main():
    file1 = Import("../data/pca_a.txt", "TAB")
    #file2 = Import("../data/pca_b.txt", "TAB")
    #file3 = Import("../data/pca_c.txt", "TAB")
    #eigen_values(file1)
    svd_function(file1)
    tsne_function(file1)
if __name__=="__main__":
    main()