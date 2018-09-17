import numpy as np


class FrequentItemsets:
    """
    Takes dataset and support as input and returns
    itemsets satisfying those criterion
    """

    def __init__(self, dataset, support):
        self.data = dataset
        self.support = support


class ImportData:
    """
    Imports data from various sources
    """

    def __init__(self, file, type):
        if type == "tab-delimited":
            self.import_tab_file(file)

    def import_tab_file(self, file):
        self.data = np.genfromtxt(file, dtype=str, delimiter='\t')

