import numpy as np
from sklearn.decomposition import TruncatedSVD
import operator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sb

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
    m1_transp = m1.transpose()
    mean_vector = np.mean(m1, axis = 0)
    centered = m1 - mean_vector
    return m1, last, centered

def eigen_values(file):
    m1, last, centered = matrix_manipulation(file)
    covprep = centered.T
    covar = np.cov(covprep)
    eig_val, eig_vec = np.linalg.eig(covar)
    eig_vec_t = eig_vec.transpose()
    eig_vec_shape = eig_vec_t.shape[1]
    eig_pairs = {}
    for i in range(eig_vec_shape):
        eig_pairs[eig_val[i]] = eig_vec[i]
    eig_matrix = sorted(eig_pairs.items(), key = lambda x: x[0], reverse = True)
    eig_matrix = eig_matrix[:2]
    return eig_matrix, last

def final_matrix(eig_matrix, file):
    m1, last, centered = matrix_manipulation(file)
    m1_transp = m1.transpose()
    #print(centered)
    data = np.concatenate((eig_matrix[0][1][:,None], eig_matrix[1][1][:,None]), axis = 1)
    plot_data = centered.dot(data)
    #print(plot_data)
    return plot_data

def svd_function(file):
    matrix, lastcol, centered = matrix_manipulation(file)
    svd = TruncatedSVD(n_components=2)
    svd_res = svd.fit_transform(matrix)
    svd_res = np.concatenate((svd_res, lastcol), axis = 1)
    #print(type(svd_res))
    return svd_res


def tsne_function(file):
    m1, lastcol, centered = matrix_manipulation(file)
    tsne = TSNE(n_components=2)
    tsne_res = tsne.fit_transform(m1)
    tsne_result = np.concatenate((tsne_res, lastcol), axis= 1)
    #print(type(tsne_result))
    return tsne_result

def plot_scatter(result, last, type):
    fig = plt.figure()
    asd = np.ravel(last)
    labels = []
    for labl in asd:
        if labl not in labels:
            labels.append(labl)
    #print(labels)
    sb.scatterplot(float(result[:,0]), float(result[:,1]), hue = asd)
    plt.xlabel('Component 1')
    plt.xlabel('Component 2')
    if type == 'pca':
        plt.title('Principle Component Analysis')
    elif type == 'svd':
        plt.title('SVD')
    elif type == 'tsna':
        plt.title('TSNE')
    plt.legend()
    plt.show()


def main():
    file1 = Import("../data/pca_a.txt", "TAB")
    file2 = Import("../data/pca_b.txt", "TAB")
    file3 = Import("../data/pca_c.txt", "TAB")

    eig_matrix, last = eigen_values(file1)
    pca_data = final_matrix(eig_matrix, file1)
    print(pca_data)
    plot_scatter(pca_data, last, 'pca')
    svd_data = svd_function(file1)
    print("-----------------")
    print(svd_data)
    plot_scatter(svd_data, last, 'svd')
    tsne_data = tsne_function(file1)
    print('-----------------')
    print(tsne_data)
    plot_scatter(tsne_data, last, 'tsne')

if __name__=="__main__":
    main()