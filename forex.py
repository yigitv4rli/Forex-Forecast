#import packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('/home/yigit/Desktop/USD_TRY4.csv',decimal=',')
df.rename(columns={"Tarih":"Date","Şimdi":"Price","Açılış":"Open","Yüksek":"High","Düşük":"Low","Fark %":"Change %"},inplace=True)

df['Date'] = pd.to_datetime(df.Date,format='%d.%m.%Y')
df.index = df['Date']

for i in range(1,5):
    ready = []
    for b in df.iloc[:,i]:
        b = '{:.2f}'.format(b)
        b = float(b)
        ready.append(b)
    df.iloc[:,i] = ready


# %% 

plt.figure(figsize=(16,8))
plt.plot(df['Open'], label='Open Price history')
plt.show()

# %% 

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
#from keras_tqdm import TQDMCallback
# , callbacks=[TQDMCallback()]

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Open'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Open'][i] = data['Open'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[:2300,:]
valid = dataset[2300:,:]

# %% 
#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Creating a data structure with 60 timesteps and one output
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# %% 
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
opening_price = model.predict(X_test)
opening_price = scaler.inverse_transform(opening_price)


# %% 

train = new_data[:2300]
valid = new_data[2300:]
valid['Predictions'] = opening_price
plt.plot(train['Open'],label='Train')
plt.plot(valid['Open'],label='True')
plt.plot(valid['Predictions'],label='Pred')
plt.legend()
plt.show()