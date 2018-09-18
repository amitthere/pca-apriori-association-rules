import numpy as np
from itertools import combinations


class FrequentItemsets:
    """
    Takes dataset and support as input and returns
    itemsets satisfying those criterion
    """

    def __init__(self, dataset, support):
        self.data = dataset
        self.support = support
        self.min_support_count = int(self.support * len(self.data) / 100)

    def frequent_1_itemsets(self):
        # Get unique items in the data with their frequency
        one_itemsets = np.unique(self.data, False, False, True, None)
        # Convert the unique items into dict, with value as frequency
        one_itemsets_dict = dict(zip(one_itemsets[0], one_itemsets[1]))
        # Use Comprehensions to get items with support about min value
        frequent_one_itemsets = {k: v for k, v in one_itemsets_dict.items() if v >= self.min_support_count}

        return list(frequent_one_itemsets.keys())

    def combinations(self, item_list):
        return



class Import:
    """
    Imports data from various sources.
    Currently, just tab-delimited files are supported
    """

    def __init__(self, file, ftype):
        self.data = None
        self.data_modified = None
        self.data_slow = None
        self.prefixed_data = None
        self.file = file
        if ftype == "TAB":
            self.import_tab_file(self.file)

    def import_tab_file(self, tabfile):
        self.data = np.genfromtxt(tabfile, dtype=str, delimiter='\t')

    def append_feature(self, feature, value):
        return 'G' + feature + '_' + value

    def process_data_1(self):
        append_feature_v = np.vectorize(self.append_feature, otypes=[str])
        self.data_modified = np.empty(self.data.shape, dtype='str')

        for col in self.data.shape[1]:
            self.data_modified[:, col] = append_feature_v(col,self.data[:, col])

        return self.data_modified

    def process_data_2(self):
        """ Slow due to using loops """
        self.data_slow = np.empty(self.data.shape, dtype='str')

        for row in range(self.data.shape[0]):
            for col in range(self.data.shape[1] - 1):
                prefix = 'G' + str(col + 1) + '_'
                self.data_slow[row][col] = prefix + str(self.data[row][col])
                # print('r:',row,' c:',col,' prefix:',prefix,' ds:',self.data_slow[row][col])

        return self.data_slow

    def process_data_3(self):
        rows = self.data.shape[0]
        cols = self.data.shape[1]
        prefix_row = ['G' + str(i) + '_' for i in range(1, cols)]
        prefix_row.append('')
        prefix_array = np.tile(prefix_row, (rows, 1))
        self.prefixed_data = np.core.defchararray.add(prefix_array, self.data)
        return self.prefixed_data




i = Import(r'F:\Google Drive\University at Buffalo\Courses'
           r'\CSE 601 - Bioinformatics and Data Mining\PA1\associationruletestdata.txt', 'TAB')
prefixed_data = i.process_data_3()

print('')