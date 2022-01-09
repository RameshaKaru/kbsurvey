import pandas as pd
from datetime import timedelta
import numpy as np
from collections import Counter
import yaml

class ProcessInformation:
    """
    This class processes the inputs
    """

    def __init__(self, cnfg):
        """
        Initializes using the configs
        """

        print("Initializing the information processing class")
        self.cnfg = cnfg
        self.df_info = self.read_information_file()

    def read_information_file(self):
        df = pd.read_csv(self.cnfg["information_file"])
        return df

    def filter_data(self):
        col_name = self.cnfg['filtering']['col_name']
        threshold = self.cnfg['filtering']['threshold']
        new_df = self.df_info[(self.df_info[col_name] >= threshold)]
        print("Filtered out", len(self.df_info)-len(new_df), "records out of", len(self.df_info))

        self.start_date_list = new_df['start_date'].nunique()
        new_df.fillna(0, inplace=True)
        self.df_info = new_df

    def get_rule_bounds(self):
        """
        Creates a list with boundaries of the rule
        :return: boundaries
                rule type
        """
        bounds = []
        rule_type = []
        rule_names = []
        for rule in self.cnfg['standard_rules']:
            p, r = self.calc_bounds(rule)
            bounds.append(p)
            rule_type.append(r)
            rule_names.append(rule['name'])
        for rule in self.cnfg['compare_rules']:
            bounds.append([0, 100])
            rule_type.append('higher')
            rule_names.append(rule['name'])

        return bounds, rule_type, rule_names

    def calc_bounds(self, rule):
        """
        Calculates the boundaries for each rule
        :param rule: dictionary with rule info
        :return: boundaries
                rule type
        """
        if rule['good_range'] == "lower":
            p = [0, 100]
            r = 'lower'
        elif rule['good_range'] == "higher":
            p = [0, 100]
            r = 'higher'
        elif rule['good_range'] == "mid":
            s = rule['range']['start']
            e = rule['range']['end']
            p = [s, e]
            r = 'mid'
        else:
            print("Rule 'good range' field is not valid")
            print(rule['name'])

        return p, r

    def get_percentages(self):
        percentages = []
        for rule in self.cnfg['standard_rules']:
            p = self.shortrule(rule)
            percentages.append(p)
        for rule in self.cnfg['compare_rules']:
            p = self.comparerule(rule)
            percentages.append(p)

        percentages = np.asarray(percentages)
        print(percentages.shape)
        print(np.min(percentages, axis=1))
        print(np.max(percentages, axis=1))
        print(np.mean(percentages, axis=1))
        self.df_info.to_csv("test.csv")

        return percentages
        # percentages_new = percentages.swapaxes(0, 1)
        # print(percentages_new.shape)
        # return percentages_new

    def shortrule(self, rule):
        print(rule['name'])
        percentages = np.asarray(self.df_info[rule['col_name']])
        return percentages

    def comparerule(self, rule):
        print("comparison rule", rule['name'])
        self.df_info[rule['name']+'_normalized'] = (self.df_info[rule['col_name']] - self.df_info[rule['col_name2']] + 100) /2
        percentages = np.asarray(self.df_info[rule['name']+'_normalized'])
        return percentages



