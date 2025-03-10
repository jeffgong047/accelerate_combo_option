from utils import Mechanism_solver_single, Market
import pandas as pd
import numpy as np

def get_counter_example():
    '''
    get the counter example for mechanism on single stock, offset=1, option_portion =1, but non frontiers got transacted 
    '''
    option_x = [1,101,3,0]
    option_y = [1,98,1000000,1]
    option_z = [1,100,2,0]
    option_u = [1,150,4.5,1]
    option_v = [1, 145,4.5,1]
    df = pd.DataFrame([option_x,option_y,option_z,option_u, option_v],columns=['C=Call, P=Put','Strike Price of the Option Times 1000','B/A_price','transaction_type'])
    return df 
counter_example = get_counter_example()
Mechanism_solver_single(Market(counter_example))