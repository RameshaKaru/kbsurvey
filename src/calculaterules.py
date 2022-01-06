import pandas as pd
from datetime import timedelta
import numpy as np
from collections import Counter
import yaml
import ast

class CalculateRules:
    """
    This class processe rules
    """

    def __init__(self, df_time_steps_list, time_steps):
        """
        Initializes
        """

        print("Initializing the rules processing class")
        self.df_time_steps_list = df_time_steps_list
        self.time_steps = time_steps
        gdf = self.calRules()

    def calRules(self):
        steps = 1
        # steps = len(self.df_time_steps_list)
        for i in range(steps):
            print(i)
            df = self.df_time_steps_list[i]
            start_date = self.time_steps[i]
            end_date = self.time_steps[i+1]

            gdf = self.qualityrules(df, start_date, end_date)
            gdf2 = self.diseases(df, start_date, end_date)



    def qualityrules(self,df, start_date, end_date):
        # df['Total_number_of_janaadhaar_id_screened'] = df.groupby(by=["family_id"])['janaadhaar_id'].transform(
            # 'nunique')

        df['duplicacy_age'] = df.groupby(by=["health_worker_id", 'religion', 'caste', 'gender', 'age'])['age'].transform('count')
        df['duplicacy_age'] = np.where((df['duplicacy_age'] > 1), 1, 0)

        # df['mobile'] = df['mobile'].map({'Yes': 1, 'No': 0})
        dict = {"Yes": 1, "No": 0}
        df = df.replace({"mobile": dict})

        gdf = df.groupby('health_worker_id').agg(
            Total_number_of_families_screened=pd.NamedAgg(column='family_id', aggfunc='nunique'),
            # Total_number_of_aadhaar_screened=pd.NamedAgg(column='aadhaar', aggfunc='nunique'),
            # Total_number_of_janaadhaar_id_screened=pd.NamedAgg(column='Total_number_of_janaadhaar_id_screened', aggfunc='nunique'),
            number_of_beneficieries=pd.NamedAgg(column='person_id', aggfunc='nunique'),
            number_of_days=pd.NamedAgg(column='date_of_survey', aggfunc='nunique'),
            number_of_houses=pd.NamedAgg(column='house_id', aggfunc='nunique'),
            duplicacy_age=pd.NamedAgg(column='duplicacy_age', aggfunc='sum'),
            mobile=pd.NamedAgg(column='mobile', aggfunc='sum'),
        )
        gdf['Average_number_of_surveys_conducted_per_day'] = gdf['number_of_beneficieries'] / gdf[
            'number_of_days']
        gdf['Average_number_of_houses_per_day'] = gdf['number_of_houses'] / gdf['number_of_days']

        # gdf['Proportion of aadhaar screened'] = gdf['Total_number_of_aadhaar_screened'] * 100 / df[
        #     'number_of_beneficieries']
        # gdf['Proportion of janaadhaar screened'] = df['Total_number_of_janaadhaar_id_screened'] * 100 / df[
        #     'Total_number_of_families_screened']

        gdf['Proportion of duplicacy name age'] = gdf['duplicacy_age'] * 100 / gdf['number_of_beneficieries']
        gdf['Proportion of mobiles'] = gdf['mobile'] * 100 / gdf['number_of_beneficieries']

        gdf['start_date'] = start_date
        gdf['end_date'] = end_date

        gdf.to_csv("test.csv")

        return gdf

    def diseases(self, df, start_date, end_date):
        df['diseases'].fillna('[\'23\']', inplace=True)
        # df['diseases_filled'] = np.where((df['duplicacy_age'] > 1), 1, 0)
        df['diseases_filled'] = df.apply(
            # lambda row: 0 if '23' in row['diseases'] else 1, axis=1)
            lambda row: 0 if any(num in row['diseases'] for num in ('22', '23')) else 1, axis=1)

        # df['diseases_list'] = [[int(idx) for idx in x.split(",")] for x in df['diseases']]
        df.diseases = df.diseases.apply(ast.literal_eval)
        # df = pd.concat([df.drop('diseases', axis=1), pd.DataFrame(df.diseases.tolist(), dtype=np.int16)], axis=1)
        # rule 4 1
        # - anemia  1
        # - malnutrition 2
        # - obesity 3
        # - diabetes 5
        # - hypertension 4
        # - lung disease 7
        # - heart disease 6
        # - kidney disease 8
        # - liver disease 9
        # - breast cancer 16
        # - visual impairment 19
        # - hearing impairment 20
        # - alcohol ?
        # - tobacco ?

        # df['diseases'] = df['diseases'].str.replace('Anemia', '1').str.replace('Malnutrition', '2').str.replace(
        #     'Obesity', '3').str.replace('Hypertension', '4').str.replace('Diabetes', '5').str.replace('Heart Disease',
        #                                                                                               '6').str.replace(
        #     'Lung Disease', '7').str.replace('Kidney Disease', '8').str.replace('Liver Disease', '9').str.replace(
        #     'Neurological Disease', '10').str.replace('Mental Health', '11').str.replace('TB', '12').str.replace('HIV',
        #                                                                                                          '13').str.replace(
        #     'Leprosy', '14').str.replace('Oral Cancer', '15').str.replace('Breast Cancer', '16').str.replace(
        #     'Cervical Cancer', '17').str.replace('Motor impairment', '18').str.replace('Visual impairment',
        #                                                                                '19').str.replace(
        #     'Hearing impairment', '20').str.replace('Other', '21').str.replace('None', '22')

        df['diseases_in_rule_4_1'] = df.apply(
            lambda row: 1 if any(num in row['diseases'] for num in (
                '1','2','3','4','5','6','7','8','9','16','19','20')) else 0, axis=1)

        # - HIV  13
        # - TB 12
        # - Leprosy 14
        # - Oral cancer 15
        # - Cervical cancer 17
        # - Motor impairment 18
        # - Neurological disease(epilepsy) 10

        df['diseases_in_rule_4_2'] = df.apply(
            lambda row: 1 if any(num in row['diseases'] for num in (
                '12', '13', '14', '15', '17', '18', '10')) else 0, axis=1)

        # - obesity 3
        # - diabetes 5
        # - heart disease 6
        # - kidney disease 8
        # - liver disease 9

        df['diseases_in_rule_4_3'] = df.apply(
            lambda row: 1 if any(num in row['diseases'] for num in (
                '3', '5', '6', '8', '9')) else 0, axis=1)

        columns = ['health_worker_id', 'person_id', 'diseases', 'diseases_filled', 'diseases_in_rule_4_1',
                   'diseases_in_rule_4_2', 'diseases_in_rule_4_3']
        df1 = pd.DataFrame(df, columns=columns)
        df1.to_csv("test1.csv")

        gdf = df.groupby('health_worker_id').agg(
            number_of_beneficieries=pd.NamedAgg(column='person_id', aggfunc='nunique'),
            number_of_days=pd.NamedAgg(column='date_of_survey', aggfunc='nunique'),
            tot_diseases_filled=pd.NamedAgg(column='diseases_filled', aggfunc='sum'),
            tot_diseases_4_1=pd.NamedAgg(column='diseases_in_rule_4_1', aggfunc='sum'),
            tot_diseases_4_2=pd.NamedAgg(column='diseases_in_rule_4_2', aggfunc='sum')
        )

        gdf['start_date'] = start_date
        gdf['end_date'] = end_date

        gdf['Proportion with diseases filled'] = gdf['tot_diseases_filled'] * 100 / gdf['number_of_beneficieries']
        gdf['Proportion_with_diseases_4_1'] = gdf['tot_diseases_4_1']*100 / gdf['number_of_beneficieries']
        gdf['Proportion_with_diseases_4_2'] = gdf['tot_diseases_4_2'] * 100 / gdf['number_of_beneficieries']


        df2 = df[df.age.notnull()]
        df2['health_worker_id'] = pd.to_numeric(df2['age'], downcast='integer')
        print(df2.age.dtype)

        df2['diseases_in_rule_4_3_more_30'] = df2.apply(
            lambda row: 1 if ((row['diseases_in_rule_4_3']) and (row['age'] > 30)) else 0, axis=1)

        df2['diseases_in_rule_4_3_less_30'] = df2.apply(
            lambda row: 1 if ((row['diseases_in_rule_4_3']) and (row['age'] <= 30)) else 0, axis=1)

        df2['age_more_30'] = df2.apply(
            lambda row: 1 if row['age'] > 30 else 0, axis=1)
        df2['age_less_30'] = df2.apply(
            lambda row: 1 if row['age'] <= 30 else 0, axis=1)

        gdf2 = df2.groupby('health_worker_id').agg(
            # number_of_beneficieries=pd.NamedAgg(column='person_id', aggfunc='nunique'),
            tot_more_30=pd.NamedAgg(column='age_more_30', aggfunc='sum'),
            tot_less_30=pd.NamedAgg(column='age_less_30', aggfunc='sum'),
            tot_diseases_4_3_more_30=pd.NamedAgg(column='diseases_in_rule_4_3_more_30', aggfunc='sum'),
            tot_diseases_4_3_less_30=pd.NamedAgg(column='diseases_in_rule_4_3_less_30', aggfunc='sum')
        )

        gdf2['Proportion_with_diseases_4_3_more_30'] = gdf2['tot_diseases_4_3_more_30'] * 100 / gdf2['tot_more_30']
        gdf2['Proportion_with_diseases_4_3_less_30'] = gdf2['tot_diseases_4_3_less_30'] * 100 / gdf2['tot_less_30']

        df_merge = pd.merge(gdf, gdf2, on='health_worker_id', how='left')
        print(len(gdf))
        print(len(gdf2))
        print(len(df_merge))

        df_merge.to_csv("test.csv")

        return df_merge






