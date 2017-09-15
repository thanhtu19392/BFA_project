
# coding: utf-8

# In[200]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
import pandas_datareader.data as web
import datetime
import seaborn as sns 
from scipy import stats
from scipy.stats import norm
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
sns.set(style = "whitegrid", color_codes = True)
sns.set(font_scale = 1)


# In[201]:


from utils.utils import get_data_frame_for_symbols, spy_data_frame, get_data_frame


# In[202]:


def compute_daily_return(data):
    daily_return = (data/data.shift(1)) - 1
    #daily_return.ix[0,:] = 0
    daily_return.fillna(0)
    return daily_return


# In[203]:


def compute_culmulative_return(data):
    cul_return = (data/data.ix[0,:].values) - 1
    return cul_return


# In[220]:


def get_stocks_stas(data_col, start, end, samples_per_year = 252,  daily_rf = 0):
    """
    Return: avg_daily_return, std_daily_return, beta, alpha, momentum, infor_ratio, sharpe_ratio
    """
    
    #data_stock = get_data_frame(symbol, start, end, dropna= True)
    daily_return = compute_daily_return(data_col)
    
    data_indice = spy_data_frame(start, end)
    indice_daily_return = compute_daily_return(data_indice)
    
    d = pd.concat([daily_return, indice_daily_return], axis = 1)
    #average, standard daily return
    column = list(d.columns)
    avg_daily_return = d[column[0]].mean()
    std_daily_return = d[column[0]].std()
    
    #covmat = np.cov(d[symbol], d["SPY"])
    covmat = np.cov(d[column[1]].fillna(0), d[column[0]].fillna(0))
    
    beta = covmat[0,1]/covmat[1,1]
    alpha= np.mean(d[column[0]])-beta*np.mean(d.SPY)
    
    #momentum
    momentum = np.prod(1 + d[column[0]].tail(12).values) -1
    
    #sharpe ratio 
    sharpe_ratio = ((d[column[0]] - daily_rf).mean()/d[column[0]].std()) * np.sqrt(samples_per_year)
    
    #information ration 
    infor_ratio = ((d[column[0]] - d.SPY).mean()/d[column[0]].std()) * np.sqrt(samples_per_year)
    
    return  avg_daily_return, std_daily_return, beta, alpha, momentum, infor_ratio, sharpe_ratio

def stas_vs_indice(portfolio_value, start, end):
    stock_stas = pd.DataFrame(list(get_stocks_stas(portfolio_value, start, end))).T
    indice_stas = pd.DataFrame(list(get_stocks_stas(get_data_frame('SPX', start, end, dropna= True), start, end))).T    
    
    stas = pd.concat([stock_stas, indice_stas])
    stas.columns = ["avg_daily_return", "std_daily_return", "beta", "alpha", "momentum", "infor_ratio", "sharpe_ratio"]
    stas.index = ["porfolio",'indice']
    return stas

# In[222]:
'''

start_date = '2017-01-01'
end_date = '2017-07-31'
list_stocks = ['AAPL','GOOG','AMZN','AXP','BAC','BA','KO','FB','IBM','GE', 'GS','HP', 'XOM', 'F','SPY']

list_stat = []
for i in range(len(list_stocks)):
    stock = list_stocks[i]
    data_col = get_data_frame(stock, start_date, end_date, dropna= True)
    stas = get_stocks_stas(data_col, start_date, end_date)
    list_stat.append(stas)

columns = ["avg_daily_return", "std_daily_return", "beta", "alpha", "momentum", "infor_ratio", "sharpe_ratio"]
df_stats = pd.DataFrame(list_stat, index = list_stocks, columns = columns).iloc[:-1,:]

'''
# In[223]:


# df_stats


# In[ ]:




