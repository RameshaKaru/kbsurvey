import time
from .processinputdata import ProcessInputData
from .calculaterules import CalculateRules





if __name__ == '__main__':
    start_time = time.time()

    processInput = ProcessInputData()
    calculaterules = CalculateRules(processInput.df_time_steps_list, processInput.time_steps)



    print("end")
    print("--- %s seconds ---" % (time.time() - start_time))
