import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


a=pd.to_datetime('2018-01-15 3:45pm')
b=pd.to_datetime('7/8/1952')
c=pd.to_datetime('7/8/1952', dayfirst=True)
print('a=',a)
print('b=',b)
print(c)


d=pd.to_datetime(['2018-01-05', '7/8/1952', 'Oct 10, 1995'])
print(d)
#DatetimeIndex(['2018-01-05', '1952-07-08', '1995-10-10'], dtype='datetime64[ns]', freq=None)

# 1) from loacl, set the right path of the file
#url='D:\Enseignement\Analyse de Données\TPs\TimeSeries/opsd_germany_daily.csv'
url='D:/Enseignement/Analyse de Données/2021 2022/TPs/TimeSeries/opsd_germany_daily.csv'

url='D:/Enseignement/Analyse de Données/2021 2022/test tp/PowerPlant.csv'

#2) dowload from Internet
#url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'
opsd_daily  = pd.read_csv(url,
                             sep=';',
                             index_col=0, # you can use the date as the index for pandas
                             parse_dates=[0]) # where is the time stamp?
print(opsd_daily.head(3))

print(opsd_daily.dtypes)

print(opsd_daily.shape)
print(opsd_daily.index)

opsd_daily['Year'] = opsd_daily.index.year
opsd_daily['Month'] = opsd_daily.index.month
#opsd_daily['Weekday Name'] = opsd_daily.index.weekday_name


#cols_plot = ['Consumption', 'Solar', 'Wind']
#axes = opsd_daily[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
#for ax in axes:
#    ax.set_ylabel('Daily Totals (GWh)')

#plt.show()


#Modele : X=Z+S1+S2+e
#data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
##1) prendre seulement la colonne Consumption
data_columns = ['Consumption']
X=opsd_daily[data_columns]
##2) Decompoistion:  X = Y+s1+e      , NB: s2 est contenue dans Y
M1 = seasonal_decompose(X, model='additive', period=7)
plt.figure("X")
M1.plot()
plt.show()

Y=M1.trend
S1=M1.seasonal
e1=M1.resid


##3) Decompoistion:  Y = Z+s2+e2      , NB: s2 est contenue dans Y
M2 = seasonal_decompose(Y, model='additive', period=365)

Y2=Y.ffill()
Y3=Y2.bfill()
M2 = seasonal_decompose(Y3, model='additive', period=365)

plt.figure("Y")
M2.plot()
plt.show()

Z=M2.trend
S2=M2.seasonal
e2=M2.resid

e=X-Z-S1-S2





Y = opsd_daily[data_columns].rolling(7, center=True).mean()   #Y=Z+S2+e'  avec variance e' faible
#estimation de Z
Z = Y[data_columns].rolling(365, center=True).mean()  # estimation de Z
#estimation de S2 sur la base de Y

from random import randrange
from pandas import Series
from statsmodels.tsa.seasonal import seasonal_decompose
series = [i+randrange(10) for i in range(1,100)]
result = seasonal_decompose(series, model='additive', period=1)
result.plot()
plt.show()






######### Affichage  Start and end of the date range to extract
start, end = '2017-01', '2017-06'
# Plot daily, weekly resampled, and 7-day rolling mean time series together
fig, ax = plt.subplots()
ax.plot(opsd_daily['Consumption'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(Y['Consumption'],
marker='.', linestyle='-', label='7-d Rolling Mean')
ax.plot(Z['Consumption'],
marker='.', linestyle='-', label='7-d Rolling Mean')

ax.set_ylabel('Consumption (GWh)')
ax.legend();
plt.show()