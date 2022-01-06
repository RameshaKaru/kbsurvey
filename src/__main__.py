import time
from .processinputdata import ProcessInputData
from .calculaterules import CalculateRules





if __name__ == '__main__':
    start_time = time.time()

    processInput = ProcessInputData()
    # processInput.read_format_raw_data()
    # calculaterules = CalculateRules(processInput.df_time_steps_list, processInput.time_steps)
    # df = calculaterules.calRules()





    print("end")
    print("--- %s seconds ---" % (time.time() - start_time))
