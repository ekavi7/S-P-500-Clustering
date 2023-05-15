#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#Downloading tickers for S&P500 from wiki
sp500=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp500_list=np.array(sp500[0]['Symbol'])
sp500_list


# In[2]:


#Taking 503 tickers to download data
get_ipython().system('pip install yfinance')
import yfinance as yf


# In[6]:


stock_list=['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ATVI', 'ADM', 'ADBE', 'ADP',
       'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE',
       'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN',
       'AMCR', 'AMD', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK',
       'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA',
       'AAPL', 'AMAT', 'APTV', 'ACGL', 'ANET', 'AJG', 'AIZ', 'T', 'ATO',
       'ADSK', 'AZO', 'AVB', 'AVY', 'BKR', 'BALL', 'BAC', 'BBWI', 'BAX',
       'BDX', 'WRB', 'BRK.B', 'BBY', 'BIO', 'TECH', 'BIIB', 'BLK', 'BK',
       'BA', 'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO',
       'BF.B', 'BG', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH',
       'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE',
       'CNC', 'CNP', 'CDAY', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG',
       'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX',
       'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CAG', 'COP',
       'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CTVA', 'CSGP', 'COST',
       'CTRA', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA',
       'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISH',
       'DIS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DD',
       'DXC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV',
       'LLY', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX',
       'EQR', 'ESS', 'EL', 'ETSY', 'RE', 'EVRG', 'ES', 'EXC', 'EXPE',
       'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX',
       'FITB', 'FRC', 'FSLR', 'FE', 'FIS', 'FISV', 'FLT', 'FMC', 'F',
       'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GEHC',
       'GEN', 'GNRC', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN',
       'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'PEAK', 'HSIC', 'HSY', 'HES',
       'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ',
       'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'ILMN', 'INCY',
       'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG',
       'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JKHY', 'J', 'JNJ', 'JCI',
       'JPM', 'JNPR', 'K', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI',
       'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS',
       'LEN', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LYB', 'MTB',
       'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH',
       'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM',
       'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ',
       'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP',
       'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN',
       'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI',
       'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OGN', 'OTIS',
       'PCAR', 'PKG', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP',
       'PKI', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'POOL',
       'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC',
       'PSA', 'PHM', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX',
       'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP',
       'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE',
       'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SNA', 'SEDG', 'SO', 'LUV',
       'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SYF', 'SNPS', 'SYY',
       'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX',
       'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG',
       'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UDR', 'ULTA', 'UNP',
       'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VRSN', 'VRSK',
       'VZ', 'VRTX', 'VFC', 'VTRS', 'VICI', 'V', 'VMC', 'WAB', 'WBA',
       'WMT', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC',
       'WRK', 'WY', 'WHR', 'WMB', 'WTW', 'GWW', 'WYNN', 'XEL', 'XYL',
       'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']


# In[7]:


data=yf.download(stock_list,start='2005-01-01',end='2019-12-31')['Adj Close']


# In[8]:


data


# In[9]:


stocks=data.dropna(axis=1)
stocks


# In[10]:


daily_returns=stocks.pct_change().dropna()
daily_returns


# In[11]:


daily_returns.describe()
#We can see that the data are more less in the same magnitude so no need for potential


# In[12]:


sp500=daily_returns.T.mean()
sp500


# In[135]:


#We put the stocks in a list so that we can use the for loop
assets=daily_returns.columns
assets

np.savetxt("MSc_stocks.csv", 
           assets,
           delimiter =", ",  # Set the delimiter as a comma followed by a space
           fmt ='% s') 
usedstocks=['A','AAP','AAPL','ABC','ABT','ACGL','ACN','ADBE','ADI','ADM','ADP','ADSK','AEE',
'AEP','AES','AFL','AIG','AIZ','AJG','AKAM','ALB','ALGN','ALK','ALL','AMAT','AMD','AME','AMGN',
'AMT','AMZN','ANSS','AON','AOS','APA','APD','APH','ARE','ATO','ATVI','AVB','AVY','AXP','AZO',
'BA','BAC','BALL','BAX','BBWI','BBY','BDX','BEN','BG','BIIB','BIO','BK','BKNG','BKR','BLK',
'BMY','BRO','BSX','BWA','BXP','C','CAG','CAH','CAT','CB','CBRE','CCI','CCL','CDNS','CHD',
'CHRW','CI','CINF','CL','CLX','CMA','CMCSA','CME','CMI','CMS','CNC','CNP','COF','COO','COP',
'COST','CPB','CPRT','CPT','CRL','CRM','CSCO','CSGP','CSX','CTAS','CTRA','CTSH','CVS','CVX',
'D','DD','DE','DGX','DHI','DHR','DIS','DISH','DLR','DLTR','DOV','DPZ','DRI','DTE','DUK','DVA',
'DVN','DXC','EA','EBAY','ECL','ED','EFX','EIX','EL','ELV','EMN','EMR','EOG','EQIX','EQR',
'EQT','ES','ESS','ETN','ETR','EVRG','EW','EXC','EXPD','EXR','F','FAST','FCX','FDS','FDX',
'FE','FFIV','FICO','FIS','FISV','FITB','FMC','FRT','GD','GE','GEN','GILD','GIS','GL',
'GLW','GOOG','GOOGL','GPC','GPN','GRMN','GS','GWW','HAL','HAS','HBAN','HD','HES','HIG','HOLX',
'HON','HPQ','HRL','HSIC','HST','HSY','HUM','IBM','IDXX','IEX','IFF','ILMN','INCY',
'INTC','INTU','IP','IPG','IRM','ISRG','IT','ITW','IVZ','J','JBHT','JCI','JKHY','JNJ',
'JNPR','JPM','K','KEY','KIM','KLAC','KMB','KMX','KO','KR','L','LEN','LH','LHX','LIN',
'LKQ','LLY','LMT','LNC','LNT','LOW','LRCX','LUV','LVS','MAA','MAR','MAS','MCD','MCHP',
'MCK','MCO','MDLZ','MDT','MET','MGM','MHK','MKC','MKTX','MLM','MMC','MMM','MNST','MO',
'MOH','MOS','MPWR','MRK','MRO','MS','MSFT','MSI','MTB','MTCH','MTD','MU','NDAQ','NDSN',
'NEE','NEM','NFLX','NI','NKE','NOC','NRG','NSC','NTAP','NTRS','NUE','NVDA','NVR','NWL',
'O','ODFL','OKE','OMC','ON','ORCL','ORLY','OXY','PAYX','PCAR','PCG','PEAK','PEG','PEP',
'PFE','PFG','PG','PGR','PH','PHM','PKG','PKI','PLD','PNC','PNR','PNW','POOL','PPG','PPL',
'PRU','PSA','PTC','PWR','PXD','QCOM','RCL','RE','REG','REGN','RF','RHI','RJF','RL','RMD',
'ROK','ROL','ROP','ROST','RSG','RTX','SBAC','SBUX','SCHW','SEE','SHW','SJM','SLB','SNA',
'SNPS','SO','SPG','SPGI','SRE','STE','STLD','STT','STX','STZ','SWK','SWKS','SYK','SYY',
'T','TAP','TDY','TECH','TER','TFC','TFX','TGT','TJX','TMO','TPR','TRMB','TROW','TRV',
'TSCO','TSN','TT','TTWO','TXN','TXT','TYL','UDR','UHS','UNH','UNP','UPS','URI','USB',
'VFC','VLO','VMC','VRSN','VRTX','VTR','VTRS','VZ','WAB','WAT','WBA','WDC','WEC','WELL',
'WFC','WHR','WM','WMB','WMT','WRB','WST','WTW','WY','WYNN','XEL','XOM','XRAY','YUM','ZBH','ZBRA','ZION']


# In[14]:


#Let's take a look at the daily_returns distibutions
import matplotlib.pyplot as plt
import seaborn as sns

for i in assets:
    plt.figure(figsize=(10,7), dpi= 80)
    sns.distplot(daily_returns[i],bins=40)


# In[15]:


#Installig the needed library for the calculation of the measures
get_ipython().system('pip install riskfolio-lib')


# In[16]:


import riskfolio as rp  


# In[23]:


#Calculation of CVaR 
def CVaR_measure_Calculation(assets):
            return rp.CVaR_Hist(daily_returns[assets],alpha=0.05)
    
CVaR_measure = map(CVaR_measure_Calculation, assets)
CVaR=list(CVaR_measure)
CVaR
#Calculation of VaR 
def VaR_measure_Calculation(assets):
            return rp.VaR_Hist(daily_returns[assets],alpha=0.05)
    
VaR_measure = map(VaR_measure_Calculation, assets)
VaR=list(VaR_measure)
VaR
#Calculation of MAD 
def MAD_measure_Calculation(assets):
            return rp.MAD(daily_returns[assets])
    
MAD_measure = map(MAD_measure_Calculation, assets)
MAD=list(MAD_measure)
MAD

#Calculation of ADD
def ADD_measure_Calculation(assets):
            return rp.ADD_Abs(daily_returns[assets])
    
ADD_measure = map(ADD_measure_Calculation, assets)
ADD=list(ADD_measure)
ADD
#Calculation of MDD
def MDD_measure_Calculation(assets):
            return rp.MDD_Abs(daily_returns[assets])
    
MDD_measure = map(MDD_measure_Calculation, assets)
MDD=list(MDD_measure)
MDD
#Calculation of UCI_Abs
w=daily_returns[assets]
def UCI_measure_Calculation(assets):
            return rp.UCI_Abs(daily_returns[assets])
    
UCI_measure = map(UCI_measure_Calculation, assets)
UCI=list(UCI_measure)
UCI
#Calculation of WR
 
def WR_measure_Calculation(assets):
            return rp.WR(daily_returns[assets])
WR_measure = map(WR_measure_Calculation, assets)
WR=list(WR_measure)
WR
#Calculation of LPM
 
def LPM_measure_Calculation(assets):
            return rp.LPM(daily_returns[assets], MAR=0, p=1)
LPM_measure = map(LPM_measure_Calculation, assets)
LPM=list(LPM_measure)
LPM
#Calculation of DaR
 
def DaR_measure_Calculation(assets):
            return rp.DaR_Abs(daily_returns[assets],alpha=0.05)
DaR_measure = map(DaR_measure_Calculation, assets)
DaR=list(DaR_measure)
DaR
#Now combining the above measures, we have the final dataset for clustering
data = pd.DataFrame(list(zip(assets, CVaR, VaR, MAD, ADD, MDD, UCI, WR, LPM, DaR)),
               columns =['assets', 'CVaR', 'VaR',  'MAD', 'ADD', 'MDD','UCI','WR','LPM','DaR'])
data


# In[193]:


data2=data.set_index(assets)
data2= data2.iloc[: , 1:]
data2


# In[194]:


import scipy as spy
import numpy as np
import pandas as pd
import seaborn as sns # For pairplots and heatmaps
import matplotlib.pyplot as plt
import scipy as spy 
def display_correlation(data2):
    r = data.corr(method="spearman")
    plt.figure(figsize=(10,6))
    heatmap = sns.heatmap(data2.corr(), vmin=-1, 
                      vmax=1, annot=True)
    plt.title("Spearman Correlation",figsize=20)
    return(r)
r=display_correlation(data2)


# In[192]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create a PCA instance: pca
pca = PCA(n_components=(9))
principalComponents = pca.fit_transform(data2)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# In[28]:


# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
PCA_components


# In[29]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
   model = KMeans(n_clusters = i, init = "k-means++")
   model.fit(PCA_components.iloc[:,:1])
   wcss.append(model.inertia_)
plt.figure(figsize=(10,10))
plt.plot(range(1,11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[30]:


import numpy as np
model = KMeans(n_clusters = 4, init = "k-means++")
label = model.fit_predict(PCA_components.iloc[:,:2])
centers = np.array(model.cluster_centers_)
uniq = np.unique(label)


# In[31]:


# colors for plotting

colors = ['red', 'green', 'orange', 'blue']
# assign a color to each features (note that we are using features as target)
features_colors = [ colors[label[i]] for i in range(len(PCA_components.iloc[:,:2])) ]
T=PCA_components.iloc[:,:2]   


# In[32]:


# plot the PCA cluster components
plt.scatter(T[0],T[1],
            c=features_colors, marker='o',
            alpha=0.4
        )
# plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            marker='x', s=100,
            linewidths=3, c=colors
        )

# store the values of PCA component in variable: for easy writing
xvector =  pca.components_[0] * max(T[0])
yvector =  pca.components_[1] * max(T[1])
columns = data.columns

# plot the 'name of individual features' along with vector length
for i in range(len(columns)):
    # plot arrows
    plt.arrow(0, 0, xvector[i], yvector[i],
                color='b', width=0.005,
                head_width=0.08, alpha=0.5
            )
    # plot name of features
    plt.text(xvector[i], yvector[i], list(columns)[i], color='b', alpha=0.75)

plt.scatter(T[0], T[1], 
            c=features_colors, marker='o',
            alpha=0.4)

#plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            marker='x', s=100,
            linewidths=3, c=colors )            
plt.show()


# In[34]:


# plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            marker='x', s=100,
            linewidths=3, c=colors
        )

# store the values of PCA component in variable: for easy writing
xvector =  pca.components_[0] * max(T[0])
yvector =  pca.components_[1] * max(T[1])
columns = data2.columns

# plot the 'name of individual features' along with vector length
for i in range(len(columns)):
    # plot arrows
    plt.arrow(0, 0, xvector[i], yvector[i],
                color='b', width=0.005,
                head_width=0.08, alpha=0.5
            )
    # plot name of features
    plt.text(xvector[i], yvector[i], list(columns)[i], color='b', alpha=0.75)

plt.scatter(T[0], T[1], 
            c=features_colors, marker='o',
            alpha=0.4)

#plot the centroids
plt.scatter(centers[:, 0], centers[:, 1],
            marker='x', s=100,
            linewidths=3, c=colors )            
plt.show()


# In[36]:


import sklearn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#Now let's find the ideal number of clusters
#elbow
wcss=[]
for i in range(1,11):
   model = KMeans(n_clusters = i, init = "k-means++")
   model.fit(data2)
   wcss.append(model.inertia_)
plt.figure(figsize=(10,10))
plt.plot(range(1,11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[37]:


#yellowbrick
get_ipython().system('pip install yellowbrick')
from yellowbrick.cluster import KElbowVisualizer
visualizer=KElbowVisualizer(model,k=(2,11))
visualizer.fit(data2)
visualizer.show()


# In[118]:


from sklearn.decomposition import PCA
pca = PCA(2)
data2 = pca.fit_transform(data2)
model = KMeans(n_clusters = 4, init = "k-means++")
label = model.fit_predict(data2)
centers = np.array(model.cluster_centers_)
uniq = np.unique(label)
plt.figure(figsize=(20, 8))
for i in uniq:
    
   plt.scatter(data2[label == i, 0] , data2[label == i , 1] , label = i)
plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')
#This is done to find the centroid for each clusters.

plt.legend(fontsize=20)
plt.title('Stock Clustering based on various risk measures',fontsize=25)


# In[105]:


label


# In[39]:


label = model.fit_predict(data2)
PerformanceData = pd.DataFrame(list(zip(assets, CVaR, VaR,  MAD, ADD, MDD, UCI, WR, LPM, DaR,label)),
               columns =['assets', 'CVaR', 'VaR',  'MAD', 'ADD', 'MDD','UCI','WR','LPM','DaR','ClusterNumber'])
PerformanceData


# In[117]:


Cluster2=PerformanceData[PerformanceData['ClusterNumber']==1]
Cluster2


# In[40]:


revert=daily_returns.T
revert
# set Index Name
revert.index.name='assets'
revert
revert['ClusterNumber'] = revert.index.map(PerformanceData.set_index('assets').ClusterNumber)
revert['ClusterNumber'] = revert['ClusterNumber'].astype(str) + '_cluster'
AddCluster=revert
AddCluster
AddCluster.sort_values(by=['ClusterNumber'])
final=AddCluster.groupby(['ClusterNumber']).mean()
final.index.name = None
final
visdata=final.T
visdata


# In[41]:


visdata['Index'] = visdata.index.map(sp500)
visdata.head()


# In[58]:


visdata.add(1).cumprod().plot(figsize=(20,8),fontsize=20)
plt.legend(fontsize=20)
plt.xlabel('Date',fontsize=20)
plt.title('Performance Comparison',fontsize=20)


# In[201]:


(visdata.expanding().mean() / visdata.expanding().std() ).tail(1000).plot(figsize=(20,8),fontsize=20)
plt.legend(fontsize=20)
plt.ylabel('Risk/Reward Ratio',fontsize=20)
plt.xlabel('Date',fontsize=20)
plt.title('Risk/Reward Comparison',fontsize=30)


# In[64]:





# In[200]:


Cluster0=PerformanceData[PerformanceData['ClusterNumber'] == 0]
Cluster0=Cluster0.set_index(Cluster0['assets'])
Cluster0= Cluster0.iloc[: , 1:]
Cluster0.to_csv('Cluster0.csv')
Cluster0


# In[199]:


Cluster2=PerformanceData[PerformanceData['ClusterNumber'] == 1]
Cluster2=Cluster2.set_index(Cluster2['assets'])
Cluster2= Cluster2.iloc[: , 1:]
Cluster2.to_csv('Cluster2.csv')
Cluster2


# In[197]:


Cluster1=PerformanceData[PerformanceData['ClusterNumber'] == 2]
Cluster1=Cluster1.set_index(Cluster1['assets'])
Cluster1= Cluster1.iloc[: , 1:]
Cluster1.to_csv('Cluster1.csv')
Cluster1


# In[198]:


Cluster3=data3[data3['ClusterNumber'] == 3]
Cluster3=Cluster3.set_index(Cluster3['assets'])
Cluster3= Cluster3.iloc[: , 1:]
Cluster3.to_csv('Cluster3.csv')
Cluster3


# In[136]:


recent=yf.download(usedstocks,start='2019-01-01',end='2023-03-31')['Adj Close']


# In[138]:


recent
revertrecent=recent.T
revertrecent


# In[140]:


# set Index Name
revertrecent.index.name='assets'
revertrecent
revertrecent['ClusterNumber'] = revertrecent.index.map(PerformanceData.set_index('assets').ClusterNumber)
revertrecent['ClusterNumber'] = revertrecent['ClusterNumber'].astype(str) + '_cluster'
AddClusterrecent=revertrecent
AddClusterrecent
AddClusterrecent.sort_values(by=['ClusterNumber'])
finalrecent=AddClusterrecent.groupby(['ClusterNumber']).mean()
finalrecent.index.name = None
finalrecent
visdatarecent=finalrecent.T
visdatarecent=visdatarecent.dropna()
visdatarecent


# In[178]:


index=recent.T.mean().dropna()
visdatarecent['Index'] = visdatarecent.index.map(index)
visdatarecent.head()


# In[184]:


visdatarecent


# In[168]:


visdatarecent.plot(figsize=(20,8),fontsize=20)
plt.legend(fontsize=20)
plt.xlabel('Date',fontsize=20),
plt.ylabel('Returns',fontsize=20)
plt.title('Recent Performance Comparison',fontsize=25)


# In[206]:


(visdatarecent.expanding().mean() / visdatarecent.expanding().std() ).tail(800).plot(figsize=(20,8),fontsize=20)
plt.legend(fontsize=20)
plt.ylabel('Risk/Reward Ratio',fontsize=20)
plt.xlabel('Date',fontsize=20)
plt.title('Risk/Reward Comparison',fontsize=25)


# In[213]:


(visdatarecent.expanding().mean() / visdatarecent.expanding().std() ).tail(400).plot(figsize=(20,8),fontsize=20)
plt.legend(fontsize=20)
plt.ylabel('Risk/Reward Ratio',fontsize=20)
plt.xlabel('Date',fontsize=20)
plt.title('Risk/Reward Comparison',fontsize=25)


# In[ ]:




