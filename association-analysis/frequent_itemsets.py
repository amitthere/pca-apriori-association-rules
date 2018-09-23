import numpy as np
from itertools import combinations


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

        self.log_frequent_itemsets(frequent_one_itemsets, k=1)

        return list(frequent_one_itemsets.keys())

    def generate_combinations(self, itemset_list, k):
        """
            Generates Combinations of k+1 itemsets
            Returns a list of sets
        """
        allset = self.merge_sets(itemset_list, k)
        if k == 1:
            fi_set = set((item,) for item in itemset_list)
        else:
            fi_set = set(frozenset(item) for item in itemset_list)
        candidate_itemsets = []

        # Approach 1
        for candidate in combinations(allset, k + 1):
            if self.is_candidate_valid(fi_set, candidate, k):
                candidate_itemsets.append(set(candidate))

        # Approach 2
        # for candidate in combinations(allset, k + 1):
        #     for s in combinations(candidate, k):
        #         if frozenset(s).issubset(fi_set) == False:
        #             break

        return candidate_itemsets

    def is_candidate_valid(self, fi_set, candidate, k):
        """ Checks if a generated candidate itemset is valid candidate"""

        subsets = set()
        valid = False

        for s in combinations(candidate, k):
            if k == 1:
                subsets.add(s)
            else:
                subsets.add(frozenset(s))

        if subsets.issubset(fi_set):
            valid = True
        return valid

    def candidate_combinations(self, itemset_list, k):
        """
        Generates Combinations of k+1 itemsets
        Returns a list of sets
        """
        candidate_itemsets = []

        for candidate in combinations(itemset_list, k + 1):
            candidate_itemsets.append(set(candidate))

        return candidate_itemsets

    def get_itemset_support(self, itemset, data):
        """ Compute support for single k-itemset """
        support = 0
        for row in range(data.shape[0]):    # try to vectorize
            if itemset.issubset(data[row]):
                support = support + 1
        return support

    def compute_support(self, itemsets):
        """
        Compute support for all itemsets with same k value.
        Returns dict of itemset and support as key and value
        """
        support = []
        for itemset in itemsets:
            s = self.get_itemset_support(itemset, self.data)
            support.append(s)
        # itemsets has set which cannot be used as keys, find a way through
        itemsets = self.set_to_str(itemsets)
        itemsets_dict = dict(zip(itemsets, support))
        return itemsets_dict


    def get_frequent_itemsets(self):
        k = 1
        fi = self.frequent_1_itemsets()

        print('\nSupport is set to ' + str(self.support) + '%')
        while len(fi) != 0:
            self.logging(k, len(fi))

            # generate candidate_combinations
            # itemsets = self.candidate_combinations(fi, k)
            itemsets = self.generate_combinations(fi, k)
            k = k + 1

            # pruning step, IMPLEMENT LATER
            # remove item-sets whose subsets are not frequent

            # get their support values
            itemsets_with_support = self.compute_support(itemsets)

            # get their frequent itemsets
            frequentitems = {k: v for k, v in itemsets_with_support.items() if v >= self.min_support_count}

            self.log_frequent_itemsets(frequentitems, k)

            fi = self.str_to_set(frequentitems.keys())
        return

    def logging(self, k, count):
        print('Number of length-' + str(k) + ' frequent itemsets: ' + str(count))

    def log_frequent_itemsets(self, fi, k):
        if len(fi) == 0:
            return
        file = r'../output/' + str(self.support) + '-' + str(k) + '-' + 'itemsets.txt'
        with open(file,'w') as f:
            for k, v in fi.items():
                f.write(k + '\t' + str(v) + '\n')
        return

    def set_to_str(self, set_list):
        str_list = [','.join(i) for i in set_list]
        return str_list

    def str_to_set(self, str_list):
        set_list = [set(i.split(',')) for i in str_list]
        return set_list

    def merge_sets(self, list, k):
        """Returns set of items from a list of sets"""
        merged_set = set()
        if k == 1:
            for s in list:
                merged_set.add(s)
        else:
            for s in list:
                merged_set.update(s)
        return merged_set



def main():
    importObject = Import(r'../data/associationruletestdata.txt', 'TAB')
    prefixed_data = importObject.process_data_3()

    support_percentage = [70, 60, 50, 40, 30]
    # support_percentage = [50]
    for support in support_percentage:
        fi = FrequentItemsets(prefixed_data, support)
        fi.get_frequent_itemsets()


if __name__ == "__main__":
    main()
