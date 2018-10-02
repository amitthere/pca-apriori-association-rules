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
    cols = file.data
    ncols = cols.shape[1]
    matrix = cols[:,0:ncols-1]
    m1 = np.array(matrix.astype(np.float))
    last_col = file.data[:,-1]
    last_col_m = np.asmatrix(last_col, dtype='U')
    last = last_col_m.transpose()
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

def plot_scatter(x, y, last, type):
    """
    Plots scatter plot for PCA, SVD and T-SNE data
    """
    fig = plt.figure()
    asd = np.ravel(last)
    labels = []
    for labl in asd:
        if labl not in labels:
            labels.append(labl)
    scatter = sb.scatterplot(x, y, hue=asd)
    plot = scatter.get_figure()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    if type == 'pca':
        plt.title('PCA - pca_c.txt')
    if type == 'svd':
        plt.title('SVD - pca_c.txt')
    elif type == 'tsne':
        plt.title('TSNE - pca_c.txt')
    plt.legend()
    plt.show()
    """if type == 'pca':
        plot.savefig('../dimensionality-reduction-plots/PCA/pca_c/PCA.png')
    if type == 'svd':
        plot.savefig('../dimensionality-reduction-plots/SVD/pca_c/SVD.png')
    if type == 'tsne':
        plot.savefig('../dimensionality-reduction-plots/T-SNE/pca_c/T-SNE.png')"""

def main():

    file0 = Import("../data/pca_c.txt", "TAB")

    eig_matrix, last = eigen_values(file0)
    pca_data = final_pca_matrix(eig_matrix, file0)
    pca_data_x = pca_data[:,0]
    pca_data_y = pca_data[:,1]
    plot_scatter(pca_data_x, pca_data_y, last, 'pca')
    svd_x, svd_y = svd_function(file0)
    plot_scatter(svd_x, svd_y, last, 'svd')
    tsne_x, tsne_y = tsne_function(file0)
    plot_scatter(tsne_x, tsne_y, last, 'tsne')

if __name__=="__main__":
    main()