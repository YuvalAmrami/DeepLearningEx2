import pandas as pd
import dataset_utils as DU
import matplotlib.pyplot as plt

stocks = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')


# stocks.sort(['symbol', 'date']
names = stocks['symbol']
google = stocks[stocks['symbol'] == 'GOOGL']
amazon = stocks[stocks['symbol'] == 'AMZN']

plt.figure()
plt.title('Google Stock')
plt.plot(google['high'])

plt.figure()
plt.title('Amazon Stock')
plt.plot(amazon['high'])

plt.show()