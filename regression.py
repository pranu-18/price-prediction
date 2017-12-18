import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

dt = quandl.get('WIKI/GOOGL')
dt = dt[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
dt['HL_PCT'] = (dt['Adj. High'] - dt['Adj. Close']) / dt['Adj. Close'] *100.0
dt['PCT_change'] = (dt['Adj. Close'] - dt['Adj. Open']) / dt['Adj. Open'] *100.0

dt = dt[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forcast_col = 'Adj. Close'
dt.fillna(-99999, inplace=True)
forcast_out = int(math.ceil(0.1*len(dt)))
dt['label'] = dt[forcast_col].shift(-forcast_out)

X = np.array(dt.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forcast_out:]
X = X[:-forcast_out]

y = dt['label']
y.dropna(inplace=True)
y = np.array(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#clf = LinearRegression(n_jobs=-1)
#clf.fit(X_train, y_train)
#with open('linearregression','wb') as f:
#	pickle.dump(clf, f)

pickle_in = open('linearregression','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forcast_set = clf.predict(X_lately)
print(forcast_set, accuracy, forcast_out)
dt['Forcast'] = np.nan

last_date = dt.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	dt.loc[next_date] = [np.nan for _ in range(len(dt.columns)-1)] + [i]

print(dt.tail())
dt['Adj. Close'].plot()
dt['Forcast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()