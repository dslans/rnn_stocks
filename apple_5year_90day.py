# LSTM Stock Price Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

## Data Management

#### Read in data
prices = pd.read_csv('data/AAPL_5yr.csv')
prices.head()

#### 90 day blocks: save last 90 days (+ extra) for test data
print("Rows: %d"%len(prices))
# [i*90 for i in range(len(prices)//90)]
test_start_index = 90*(len(prices)//90 - 1)
train_df = prices[:test_start_index]
test_df = prices[test_start_index:]
print('\nTrain Data \n-------- \n\
Rows: %d \nStart Date: %s \nEnd Date: %s' \
% (len(train_df),min(train_df.Date),max(train_df.Date)))
print('\nTest Data \n-------- \n\
Rows: %d \nStart Date: %s \nEnd Date: %s' \
% (len(test_df),min(test_df.Date),max(test_df.Date)))


#### Normalize data
predictors = ['Close','Volume']
scaler = MinMaxScaler() # scale 0 to 1
train = train_df[predictors].to_numpy().reshape(-1,len(predictors))
train = scaler.fit_transform(train)
train.shape

#### Create Inputs and Outputs

# 180 days of prices to predict price 90 days in the future
hist_days = 180
future_days = 30
future_index = hist_days + future_days
output_data = train[future_index:,0]
input_data = np.array([train[i:i+hist_days] for i in \
range(len(output_data))])

# input_data, output_data = [], []
# for day in range(900):
#     indata = train[day:day+90]
#     outdata = train[day+90:day+180][-1][0]
#     input_data.append(indata), output_data.append(outdata)

# input_data, output_data = np.array(input_data), np.array(output_data)
input_data.shape
# output_data = output_data.reshape(1,-1)
output_data.shape

## Predict with LSTM

#### Build the LSTM

# initialize with a sequential model
model = Sequential()

# add 1st layer and dropout
model.add(LSTM(units=100, return_sequences=True, \
input_shape=(input_data.shape[1],input_data.shape[2])))
model.add(Dropout(0.2))

# add 2nd layer
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

# add 3rd layer
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

# add 4th layer
model.add(LSTM(units=100))
model.add(Dropout(0.2))

# add Dense layer
model.add(Dense(units=1)) # units=1 for prediction of a single value

# compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define callbacks
es = EarlyStopping(monitor='loss', mode='min', verbose=1)
# model_checkpoint = ModelCheckpoint()

# fit the model
history = model.fit(input_data, output_data, epochs=20, batch_size=32, callbacks=[es])

#### Prepare test data
model.history.history

train_test = pd.concat([train_df, test_df], axis=0)
test_all = train_test[test_df.index.start - (price_hist_days + future_price_days):][predictors].to_numpy()

# test_all = test_all.reshape(-1,1)
test_all = scaler.fit_transform(test_all)
test_all.shape

test_output_data = test_all[first_future_price:,0]
test_input_data = np.array([test_all[i:i+price_hist_days] for i in \
range(len(test_output_data))])
test_input_data.shape
test_output_data.shape

predictions = model.predict(test_input_data)

test_all.shape
predictions.shape
preds_all_cols = np.concatenate((predictions,np.zeros(len(predictions)).reshape(-1,1)), axis=1)


predictions_unscale = scaler.inverse_transform(preds_all_cols)[:,0]


#### plot predictions
predicted_price = pd.DataFrame(predictions_unscale, \
index=range(test_df.index.start, test_df.index.stop), columns=['Close'])
plt.figure(figsize=(10,6))
plt.plot(train_test.Close[900:], color='blue', label='Actual Price')
plt.plot(predicted_price.Close, color='red', label='Predicted Price')
plt.xticks(range(train_test.index.start + 900, train_test.index.stop,30), \
train_test['Date'].loc[::train_test.index.stop], rotation=45)
plt.legend()
plt.show()

import shap

%%time
explainer = shap.DeepExplainer(model, input_data[:100])


%%time
shap_values = explainer.shap_values(test_input_data[:10])


shap.initjs()

explainer.expected_value[0]

shap.force_plot(explainer.expected_value[0], shap_values[0][0])


len(shap_values[0][0])
