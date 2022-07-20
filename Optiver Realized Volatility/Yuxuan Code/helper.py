import pandas as pd
import numpy as np
from enum import Enum
import os
from IPython.core.debugger import set_trace

DATA_DIR = '/Users/yuxuan/Desktop/kaggle_optiver/optiver-realized-volatility-prediction'

# def load_stock_data(stock_id: int, directory: str) -> pd.DataFrame:
#     return pd.read_parquet(os.path.join(DATA_DIR, directory, f'stock_id={stock_id}'))

# def load_data(stock_id: int, stem: str, split: str) -> pd.DataFrame:
#     if split == 'train':
#         return load_stock_data(stock_id, f'{stem}_train.parquet')
#     elif split == 'test':
#         return load_stock_data(stock_id, f'{stem}_test.parquet')
#     else:
#         return pd.concat([
#             load_data(stock_id, stem, 'train'),
#             load_data(stock_id, stem, 'test')
#         ]).reset_index(drop=True)
    
# def load_book(stock_id: int, split = 'train') -> pd.DataFrame:
#     return load_data(stock_id, 'book', split)

# def load_trade(stock_id: int, split = 'train') -> pd.DataFrame:
#     return load_data(stock_id, 'trade', split)


class DataBlock(Enum):
    TRAIN = 1
    TEST = 2
    BOTH = 3
    
def load_stock_data(stock_id: int, directory: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(DATA_DIR, directory, f'stock_id={stock_id}'))

def load_data(stock_id: int, stem: str, block: DataBlock) -> pd.DataFrame:
    if block == DataBlock.TRAIN:
        return load_stock_data(stock_id, f'{stem}_train.parquet')
    elif block == DataBlock.TEST:
        return load_stock_data(stock_id, f'{stem}_test.parquet')
    else:
        return pd.concat([
            load_data(stock_id, stem, DataBlock.TRAIN),
            load_data(stock_id, stem, DataBlock.TEST)
        ]).reset_index(drop=True)

def load_book(stock_id: int, block: DataBlock=DataBlock.TRAIN) -> pd.DataFrame:
    return load_data(stock_id, 'book', block)


def load_trade(stock_id: int, block=DataBlock.TRAIN) -> pd.DataFrame:
    return load_data(stock_id, 'trade', block)

#########

def get_wap(bid_p, bid_s, ask_p, ask_s):
    return (bid_p*ask_s+ask_p*bid_s)/(bid_s+ask_s)

def get_wap_from_df(df, bid_p:str, bid_s:str, ask_p:str, ask_s:str):
    l = locals()
    names = [l[w1+w2] for w1 in ['bid', 'ask'] for w2 in ['_p', '_s']]
    inputs = [df[name] for name in names]
    #print(names)
    return get_wap(*inputs)

def get_log_return_from_df(df, bid_p:str, bid_s:str, ask_p:str, ask_s:str):
    wap = get_wap_from_df(**locals())
    log_r = wap.groupby(df['time_id']).apply(log_return)
    return log_r.dropna()
    
def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    '''SD under zero mean
    '''
    return np.sqrt(np.sum(series_log_return**2))

def log_return_per_time_id(file_path):
    df_book_data = pd.read_parquet(file_path)
    df_book_data['wap'] = get_wap(bid_p = df_book_data['bid_price1'], 
                                  bid_s = df_book_data['bid_size1'], 
                                  ask_p = df_book_data['ask_price1'], 
                                  ask_s = df_book_data['ask_size1'])
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    # drop 0 seconds_in_bucket 
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]
    return df_book_data

#

def get_working_data(seed = 100, train_ratio = 0.8, train = True):
    all_target = pd.read_csv(f'{DATA_DIR}/train.csv')
    stock_ids = all_target.stock_id.unique()
    #set_trace()
    # generate test
    rng = np.random.default_rng(seed)
    #n_train = len(train_target)
    # option 1: 
    #train_row = rng.integers(0, n_train, size = int(n_train * 0.2))
    # option 2:
    train_ids = rng.choice(stock_ids, size = int(len(stock_ids) * train_ratio), replace = False)
    #set_trace()
    row_in_train = all_target.stock_id.apply(lambda x: x in train_ids)
    if train:
        train_target = all_target[row_in_train]
        return {'data':train_target, 'ids':train_ids}
    else:
        test_ids = [x for x in stock_ids if x not in train_ids]
        test_target = all_target[~row_in_train]
        return {'data':test_target, 'ids':test_ids}
