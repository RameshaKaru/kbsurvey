import time
from .processinputdata import ProcessInputData
from .calculaterules import CalculateRules
from .processinformation import ProcessInformation
from .kde import KDEs
from .scores import Scores



if __name__ == '__main__':
    start_time = time.time()

    processInput = ProcessInputData()
    # processInput.read_format_raw_data()
    # calculaterules = CalculateRules(processInput.df_time_steps_list, processInput.time_steps)
    # df = calculaterules.calRules()

    processInfo = ProcessInformation(processInput.cnfg)
    processInfo.filter_data()
    bounds, rule_type, rule_names = processInfo.get_rule_bounds()
    p = processInfo.get_percentages()
    percentages = p.tolist()

    # draw KDES
    kdecls = KDEs()
    kdes = kdecls.func_get_all_kdes(rule_type, bounds, percentages, rule_names)

    # get non diligence scores
    sc = Scores(kdes=kdes, rule_type=rule_type, bounds=bounds,
                func_get_prob_mass_trans=kdecls.func_get_prob_mass_trans, df=processInfo.df_info)
    fraud_prob = sc.get_fraud_probs(p.T)
    norm_scores = sc.get_simple_norm(fraud_prob)







    print("end")
    print("--- %s seconds ---" % (time.time() - start_time))
