import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

class Scores:

    def __init__(self, kdes, rule_type, bounds, func_get_prob_mass_trans, df):
        print("Initializing the non diligence score calculation class")
        self.kdes = kdes
        self.rule_type = rule_type
        self.bounds = bounds
        self.func_get_prob_mass_trans = func_get_prob_mass_trans
        self.df = df

    def rule_prob(self, p, r):
        probabilities = []
        if self.rule_type[r] == 'lower':
            for x in p:
                x = float(x)
                prob = self.func_get_prob_mass_trans(self.kdes[r], 0, x)
                probabilities.append(prob[0])
            return np.asarray(probabilities)

        elif self.rule_type[r] == 'higher':
            for x in p:
                x = float(x)
                prob = self.func_get_prob_mass_trans(self.kdes[r], x, 100)
                probabilities.append(prob[0])
            return np.asarray(probabilities)

        elif self.rule_type[r] == 'mid':
            s = self.bounds[r][0]
            e = self.bounds[r][1]
            for x in p:
                x = float(x)
                if (x < s):
                    probR = self.func_get_prob_mass_trans(self.kdes[r][0], x, s)
                    prob = probR[0]
                elif (x > e):
                    probR = self.func_get_prob_mass_trans(self.kdes[r][1], e, x)
                    prob = probR[0]
                else:
                    prob = 0.0
                probabilities.append(prob)
            return np.asarray(probabilities)

    def get_fraud_probs(self, percentages):
        fraud_prob = np.zeros(percentages.shape)

        for r in range(percentages.shape[1]):
            fraud_prob[:, r] = self.rule_prob(percentages[:, r], r)

        return fraud_prob

    def get_simple_norm(self, prob):
        norms = np.linalg.norm(prob, axis=1, ord=2)/10

        nsq = np.power(prob,2)
        hw_score = np.zeros(len(prob))
        hw_ben_score = np.zeros(len(prob))
        for i in range(10):
            if (i == 0) or (i == 1):
                hw_score = hw_score + nsq[:, i]
            else:
                hw_ben_score = hw_ben_score + nsq[:,i]

        hw_score = np.power(hw_score,0.5)/2
        hw_ben_score = np.power(hw_ben_score, 0.5)/8

        self.df['scores'] = norms
        self.df['health_worker_dependant_scores'] = hw_score
        self.df['beneficiery_and_health_worker_dependant_scores'] = hw_ben_score
        self.df.to_csv("rules/raw_scores.csv")
        columns = ['health_worker_id', 'start_date', 'end_date', 'scores', 'health_worker_dependant_scores',
                   'beneficiery_and_health_worker_dependant_scores']
        new_df = pd.DataFrame(self.df, columns=columns)
        new_df.to_csv("rules/scores.csv")

        print("Correlation between scores & hw scores:", pearsonr(norms, hw_score)[0])
        print("Correlation between scores & hw+ben scores:", pearsonr(norms, hw_ben_score)[0])
        print("Correlation between hw & hw+ben scores:", pearsonr(hw_score, hw_ben_score)[0])

        return norms