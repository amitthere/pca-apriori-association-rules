import numpy as np
from sklearn.decomposition import TruncatedSVD
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
    """
    Takes the file and
    returns the matrix for which you find covariance
    and eigen values and eigen vectors, normalized matrix and last column
    for labels in scatter plot
    """
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
    """
    Takes the file and returns eigen matrix
    that contains eigen values and eigen vectors
    sorted according to eigen values
    """
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

def final_pca_matrix(eig_matrix, file):
    """
    Returns the final PCA data to be plotted
    """
    m1, last, centered = matrix_manipulation(file)
    m1_transp = m1.transpose()
    data = np.concatenate((eig_matrix[0][1][:,None], eig_matrix[1][1][:,None]), axis = 1)
    plot_data = centered.dot(data)
    return plot_data

def svd_function(file):
    """
    Uses TruncatedSVD function from sklearn package
    and returns svd data to be plotted
    """
    matrix, lastcol, centered = matrix_manipulation(file)
    svd = TruncatedSVD(n_components=2)
    svd_res = svd.fit_transform(matrix)
    svd_res = np.concatenate((svd_res, lastcol), axis = 1)
    svd_1 = svd_res[:,0]
    svd_2 = svd_res[:,1]
    svd_x1 = np.ravel(svd_1)
    svd_y1 = np.ravel(svd_2)
    svd_x = svd_x1.astype(float)
    svd_y = svd_y1.astype(float)
    return svd_x, svd_y


def tsne_function(file):
    """
    Uses TSNE from sklearn package and
    returns tsne data to be plotted
    """
    m1, lastcol, centered = matrix_manipulation(file)
    tsne = TSNE(n_components=2)
    tsne_res = tsne.fit_transform(m1)
    tsne_result = np.concatenate((tsne_res, lastcol), axis= 1)
    #print(tsne_result)
    tsne_1 = tsne_result[:,0]
    tsne_2 = tsne_result[:,1]
    tsne_x1 = np.ravel(tsne_1)
    tsne_y1 = np.ravel(tsne_2)
    tsne_x = tsne_x1.astype(float)
    tsne_y = tsne_y1.astype(float)
    return tsne_x, tsne_y

def plot_scatter(result, last):
    """
    Plots a scatter plot for PCA data
    """
    fig = plt.figure()
    asd = np.ravel(last)
    labels = []
    for labl in asd:
        if labl not in labels:
            labels.append(labl)
    sb.scatterplot(result[:,0], result[:,1], hue = asd)
    plt.xlabel('Component 1')
    plt.xlabel('Component 2')
    plt.title('Principle Component Analysis')
    plt.legend()
    plt.show()

def plot_scatter_svd_tsne(x, y, last, type):
    """
    Plots scatter plot for SVD and TSNE data
    """
    fig = plt.figure()
    asd = np.ravel(last)
    labels = []
    for labl in asd:
        if labl not in labels:
            labels.append(labl)
    sb.scatterplot(x, y, hue=asd)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    if type == 'svd':
        plt.title('SVD')
    elif type == 'tsne':
        plt.title('TSNE')
    plt.legend()
    plt.show()

def main():
    file1 = Import("../data/pca_a.txt", "TAB")
    file2 = Import("../data/pca_b.txt", "TAB")
    file3 = Import("../data/pca_c.txt", "TAB")

    eig_matrix, last = eigen_values(file1)
    pca_data = final_pca_matrix(eig_matrix, file1)
    plot_scatter(pca_data, last)
    svd_x, svd_y = svd_function(file1)
    plot_scatter_svd_tsne(svd_x, svd_y, last, 'svd')
    tsne_x, tsne_y = tsne_function(file1)
    plot_scatter_svd_tsne(tsne_x, tsne_y, last, 'tsne')

if __name__=="__main__":
    main()