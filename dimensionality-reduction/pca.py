import numpy as np

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

file1 = Import("../data/pca_a.txt", "TAB")
file2 = Import("../data/pca_b.txt", "TAB")
file3 = Import("../data/pca_c.txt", "TAB")

def normalize_matrix(self, file):
    matrix1 = file1.data[:,0:4]
    m1 = np.array(matrix1.astype(np.float))
    #print(m1)

    mean_vector1 = np.mean(m1, axis= 0)
    #print(mean_vector1)

    centered1 = mean_vector1 - m1
    print(centered1)