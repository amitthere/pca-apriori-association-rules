import numpy as np
import copy
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
        self.frequent_itemsets_1 = None
        self.all_frequent_itemsets = None
        self.min_support_count = int(self.support * len(self.data) / 100)

    def frequent_1_itemsets(self):
        # Get unique items in the data with their frequency
        one_itemsets = np.unique(self.data, False, False, True, None)

        # Convert the unique items into dict, with value as frequency
        one_itemsets_dict = dict(zip(one_itemsets[0], one_itemsets[1]))

        # Use Comprehensions to get items with support about min value
        frequent_one_itemsets = {k: v for k, v in one_itemsets_dict.items() if v >= self.min_support_count}

        self.log_frequent_itemsets(frequent_one_itemsets, k=1)
        self.frequent_itemsets_1 = frequent_one_itemsets
        self.all_frequent_itemsets = frequent_one_itemsets

        return self.frequent_itemsets_1

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
            self.all_frequent_itemsets = dict(self.all_frequent_itemsets, **frequentitems)

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


class AssociationRules:

    def __init__(self, confidence, frequent_itemsets):
        self.confidence = confidence
        self.fi = frequent_itemsets
        self.rules = []

    def template1(self, part, count, items):
        result = []
        if part == 'RULE':
            if count == 'ANY':
                for rule in self.rules:
                    for item in items:
                        sitem = set((item,))
                        if sitem.issubset(rule[0]) or sitem.issubset(rule[1]):
                            result.append(rule)
                            break
            elif count == 'NONE':
                result = copy.deepcopy(self.rules)
                for rule in self.rules:
                    for item in items:
                        sitem = set((item,))
                        if sitem.issubset(rule[0]) or sitem.issubset(rule[1]):
                            result.remove(rule)
                            break
            elif count == '1':
                for rule in self.rules:
                    for item in items:
                        sitem = set((item,))
                        if sitem.issubset(rule[0]) or sitem.issubset(rule[1]):
                            temp_items = copy.deepcopy(items)
                            temp_items.remove(item)

                            # check if rest of the items are in the rule
                            for ri in temp_items:
                                ritem = set((ri,))
                                if ritem.issubset(rule[0]) or ritem.issubset(rule[1]):
                                    break
                            else:   # this else block is ONLY executed if above for loop did NOT break
                                result.append(rule)
                                continue
                            break   # this executes ONLY if inner loop DID break,
        elif part == 'HEAD':
            if count == 'ANY':
                for rule in self.rules:
                    for item in items:
                        sitem = set((item,))
                        if sitem.issubset(rule[0]):
                            result.append(rule)
                            break
            elif count == 'NONE':
                result = copy.deepcopy(self.rules)
                for rule in self.rules:
                    for item in items:
                        sitem = set((item,))
                        if sitem.issubset(rule[0]):
                            result.remove(rule)
                            break
            elif count == '1':
                for rule in self.rules:
                    for item in items:
                        sitem = set((item,))
                        if sitem.issubset(rule[0]):
                            temp_items = copy.deepcopy(items)
                            temp_items.remove(item)

                            # check if rest of the items are in the rule
                            for ri in temp_items:
                                ritem = set((ri,))
                                if ritem.issubset(rule[0]):
                                    break
                            else:   # this else block is ONLY executed if above for loop did NOT break
                                result.append(rule)
                                continue
                            break

        elif part == 'BODY':
            if count == 'ANY':
                for rule in self.rules:
                    for item in items:
                        sitem = set((item,))
                        if sitem.issubset(rule[1]):
                            result.append(rule)
                            break
            elif count == 'NONE':
                result = copy.deepcopy(self.rules)
                for rule in self.rules:
                    for item in items:
                        sitem = set((item,))
                        if sitem.issubset(rule[1]):
                            result.remove(rule)
                            break
            elif count == '1':
                for rule in self.rules:
                    for item in items:
                        sitem = set((item,))
                        if sitem.issubset(rule[1]):
                            temp_items = copy.deepcopy(items)
                            temp_items.remove(item)

                            # check if rest of the items are in the rule
                            for ri in temp_items:
                                ritem = set((ri,))
                                if ritem.issubset(rule[1]):
                                    break
                            else:   # this else block is ONLY executed if above for loop did NOT break
                                result.append(rule)
                                continue
                            break

        return result, len(result)

    def template2(self, part, size):
        result = []
        if part == 'RULE':
            for rule in self.rules:
                itemset = rule[0].union(rule[1])
                if len(itemset) >= int(size):
                    result.append(rule)
        elif part == 'HEAD':
            for rule in self.rules:
                if len(rule[0]) >= int(size):
                    result.append(rule)
        elif part == 'BODY':
            for rule in self.rules:
                if len(rule[1]) >= int(size):
                    result.append(rule)
        return result, len(result)

    def template3(self, type, *args):
        result = []
        if type == '1or1':
            O1, C1 = self.template1(args[0], args[1], args[2])
            O2, C2 = self.template1(args[3], args[4], args[5])
            result = O1 + [i for i in O2 if i not in O1]
        elif type == '1and1':
            O1, C1 = self.template1(args[0], args[1], args[2])
            O2, C2 = self.template1(args[3], args[4], args[5])
            result = [i for i in O1 if i in O2]
        elif type == '1or2':
            O1, C1 = self.template1(args[0], args[1], args[2])
            O2, C2 = self.template2(args[3], args[4])
            result = O1 + [i for i in O2 if i not in O1]
        elif type == '1and2':
            O1, C1 = self.template1(args[0], args[1], args[2])
            O2, C2 = self.template2(args[3], args[4])
            result = [i for i in O1 if i in O2]
        elif type == '2or2':
            O1, C1 = self.template2(args[0], args[1])
            O2, C2 = self.template2(args[2], args[3])
            result = O1 + [i for i in O2 if i not in O1]
        elif type == '2and2':
            O1, C1 = self.template2(args[0], args[1])
            O2, C2 = self.template2(args[2], args[3])
            result = [i for i in O1 if i in O2]
        return result, len(result)

    def get_itemset_support(self, itemset, data):
        """ Compute support for single k-itemset """
        support = 0
        for row in range(data.shape[0]):
            if itemset.issubset(data[row]):
                support = support + 1
        return support

    def generate_rules(self):
        fi_g2 = {k:v for k,v in self.fi.all_frequent_itemsets.items() if k not in self.fi.frequent_itemsets_1}

        print('\nConfidence is set to : ' + str(self.confidence) + '%')
        # Rule is : HEAD -> BODY

        # iterate through all frequent itemsets of length > 1
        for itemset in fi_g2:

            # get all 1-itemsets in the 'itemset' of length > 1
            all_items = itemset.split(',')

            # for each 1-itemset, do sequentially
            for index in range(1, len(all_items)):

                # get all subsets of length 1 to len(itemset) in this for loop
                subsets = set(combinations(all_items, index))

                for subset in subsets:

                    num = self.get_itemset_support(set(all_items), self.fi.data)
                    deno = self.get_itemset_support(set(subset), self.fi.data)
                    confidence = float(num)/float(deno)

                    # if rule has enough confidence, add it to the set of rules
                    if (confidence*100) >= self.confidence:
                        r_head = set(subset)                        # X
                        r_body = (set(all_items) - set(subset))     # Y
                        rule = [r_head, r_body]                     # X -> Y - X
                        self.rules.append(rule)

        return

    def read_template1_args(self):
        part = input('Enter argument 1 :')
        if part not in ['RULE', 'BODY', 'HEAD']:
            print('Please enter correct values.')
            return None
        count = input('Enter argument 2 :')
        if count not in ['ANY', 'NONE', '1']:
            print('Please enter correct values.')
            return None
        items = input('Enter argument 3 :')
        items = items.split()
        vitems = []
        for i in items:
            s = i.split('_')
            if len(s) > 1:
                n = s[0].upper() + '_' + s[1][0].upper() + s[1][1:].lower()
                vitems.append(n)
            else:
                vitems.append(i)
        return part,count,vitems

    def read_template2_args(self):
        part = input('Enter argument 1 :')
        if part not in ['RULE', 'BODY', 'HEAD']:
            print('Please enter correct values.')
            return None
        size = input('Enter argument 2 :')
        if size.isdigit() == False:
            print('Please enter correct values.')
            return None
        return part, size

    def template_queries(self):

        while True:
            template = input('Enter Template Number (0 to exit) :')
            if template == '0':
                break
            elif template == '1':
                try:
                    part, count, items = self.read_template1_args()
                except:
                    continue

                rules, number = self.template1(part, count, items)
                self.print_query_output(rules, number)
            elif template == '2':
                try:
                    part, size = self.read_template2_args()
                except:
                    continue

                rules, number = self.template2(part, size)
                self.print_query_output(rules, number)
            elif template == '3':
                type = input('Enter argument 1 :')
                try:
                    if type not in ['1or1', '1and1', '1or2', '1and2', '2or2', '2and2']:
                        print('Please enter correct values.')
                        continue
                    elif type == '1or1' or type == '1and1':
                        arg0, arg1, arg2 = self.read_template1_args()
                        arg3, arg4, arg5 = self.read_template1_args()
                        rules, number = self.template3(type, arg0, arg1, arg2, arg3, arg4, arg5)
                    elif type == '1or2' or type == '1and2':
                        arg0, arg1, arg2 = self.read_template1_args()
                        arg3, arg4  = self.read_template2_args()
                        rules, number = self.template3(type, arg0, arg1, arg2, arg3, arg4)
                    elif type == '2or2' or type == '2and2':
                        arg0, arg1 = self.read_template2_args()
                        arg2, arg3 = self.read_template2_args()
                        rules, number = self.template3(type, arg0, arg1, arg2, arg3)
                except:
                    print('Please enter correct values.')
                    continue
                self.print_query_output(rules, number)
        return

    def print_query_output(self, rules, number):
        for rule in rules:
            print(str(rule[0]) + '\t->\t' + str(rule[1]))
        print('Total rules for this query : ' + str(number))
        return

    def log_rules(self):
        file = r'../output/rules-support-' + str(self.fi.support) + '-confidence-' + str(self.confidence) + '.txt'
        with open(file, 'w') as f:
            for rule in self.rules:
                f.write(str(rule[0]) + '\t->\t' + str(rule[1]) + '\n')
        return



def main():
    importObject = Import(r'associationruletestdata.txt', 'TAB')
    prefixed_data = importObject.process_data_3()

    # take support as user input
    # support_percentage = [70, 60, 50, 40, 30]
    support_percentage = [int(input("Enter minimum support score: "))]
    frequentitems = []
    for support in support_percentage:
        fi = FrequentItemsets(prefixed_data, support)
        fi.get_frequent_itemsets()
        frequentitems.append(fi)
        print('Number of all lengths frequent itemsets: ' + str(len(fi.all_frequent_itemsets)) + '\n')

    # for f in frequentitems:
    #     print('For support of '+ str(f.support) + '% total no of frequent itemsets : ' + str(len(f.all_frequent_itemsets)))

    # take confidence as user input
    confidence = int(input("Enter minimum confidence score: "))
    ar = AssociationRules(confidence, frequentitems[0])
    ar.generate_rules()
    ar.log_rules()
    print('Total Number of Rules generated : ', len(ar.rules))

    ar.template_queries()

    return


if __name__ == "__main__":
    main()
