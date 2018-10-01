import numpy as np
from itertools import combinations

class AssociationRules:

    def __init__(self, confidence, frequent_itemsets):
        self.confidence = confidence
        self.fi = frequent_itemsets
        self.rules = {}

    def template1(self, arg1, arg2, arg3):
        return

    def template2(self, arg1, arg2):
        return

    def template3(self, type, *args):
        return

    def get_itemset_support(self, itemset, data):
        """ Compute support for single k-itemset """
        support = 0
        for row in range(data.shape[0]):
            if itemset.issubset(data[row]):
                support = support + 1
        return support

    def generate_rules(self):
        fi_g2 = {k:v for k,v in self.fi.all_frequent_itemsets.items() if k not in self.fi.frequent_itemsets_1}

        # Rule is : BODY --> HEAD
        BODY = []
        HEAD = []

        # iterate through all frequent itemsets of length > 1
        for itemset in fi_g2:

            # get all 1-itemsets in the 'itemset' of length > 1
            all_items = list(itemset)

            # for each 1-itemset, do sequentially
            for index in range(1, len(all_items)):

                # get all subsets of length 1 to len(itemset)
                subsets = set(combinations(all_items, index))

                for subset in subsets:

                    # option 1
                    confidence = self.get_itemset_support(itemset, self.fi.data)/float(self.get_itemset_support(set(subset), self.fi.data))

                    # option 2
                    # subset is BODY here and (itemset - subset) is HEAD
                    # since both must be frequent, get their support from all_frequent_itemsets dictionary

                    # if rule has enough confidence, add it to the set of rules
                    if confidence >= (self.confidence/100.0):
                        r_body = subset
                        r_head = (set(all_items) - set(subset))
                        self.rules[r_body] = r_head
                        BODY.append(r_body)
                        HEAD.append(r_head)
                        print('RULE: '+ str(r_body) + ' -> ' + str(r_head))






def main():
    return


if __name__ == "__main__":
    main()

