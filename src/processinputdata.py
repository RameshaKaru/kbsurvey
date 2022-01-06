import pandas as pd
from datetime import timedelta
import numpy as np
from collections import Counter
import yaml

class ProcessInputData:
    """
    This class processes the inputs
    """

    def __init__(self):
        """
        Initializes using the configs
        """

        print("Initializing the input processing class")
        self.cnfg = self.get_configs()
        self.df = self.read_data()
        self.df = self.format_df()
        self.df_time_steps_list, self.time_steps = self.divide_time_steps()
        # print("test", Counter(self.df['mobile']))



    def get_configs(self):
        """
        Obtains the configurations from the config file

        Returns:
            dictionary with configurations

        """

        with open('config.yaml', 'r') as stream:
            try:
                configs = yaml.safe_load(stream)
                return configs
            except yaml.YAMLError as exc:
                print(exc)

    def read_data(self):

        num_files = self.cnfg['num_files']
        dfs = []

        for i in range(1, num_files + 1):
            df = pd.read_csv(self.cnfg["location"] + str(i) + ".csv", sep=',',
                             error_bad_lines=False, index_col=False, dtype='unicode')
            # print("total", i, len(df))
            dfs.append(df)

        tot_df = pd.concat(dfs)
        print("total records", len(tot_df))
        print("Total health workers: ", tot_df["health_worker_id"].nunique())
        print("Total districts: ", tot_df["district_id"].nunique())
        print("Total sub centers (ANMs): ", tot_df["sub_center_id"].nunique())
        print("Total villages: ", tot_df["village_id"].nunique())
        print("Total families: ", tot_df["family_id"].nunique())

        return tot_df

    def format_df(self):
        tot_df = self.df
        tot_df['health_worker_id'] = pd.to_numeric(tot_df['health_worker_id'], downcast='integer')
        tot_df['date_of_survey'] = pd.to_datetime(tot_df['date_of_survey'])
        # print("Check", tot_df['age'].isna().sum())
        # print(tot_df.age.dtype)

        temp = tot_df[(tot_df['date_of_survey'] > '2021-1-1')]
        return temp

    def divide_time_steps(self):
        print("Survey duration:", self.df['date_of_survey'].min(), self.df['date_of_survey'].max())

        period = self.cnfg["period"]
        df_time_steps = []
        time_steps = []
        points_list = []
        start_date = self.df['date_of_survey'].min()
        last_date = self.df['date_of_survey'].max()
        end_date = start_date + timedelta(days=period)
        time_steps.append(start_date)
        print(end_date)

        while start_date <= last_date:
            temp = self.df[(self.df['date_of_survey'] >= start_date) & (self.df['date_of_survey'] < end_date)]
            df_time_steps.append(temp)
            time_steps.append(end_date)
            points_list.append(temp["health_worker_id"].unique())
            print(start_date, end_date, len(temp["health_worker_id"].unique()))
            start_date = end_date
            end_date = end_date + timedelta(days=period)

        flat_list = [item for sublist in points_list for item in sublist]
        c = Counter(flat_list)
        print("Datapoints (1 week gap):", Counter(c.values()))
        data_points = pd.DataFrame({"health_worker_id": c.keys(), "num_datapoints": c.values()})
        data_points.to_csv("datapoints_count.csv")

        return df_time_steps, time_steps




