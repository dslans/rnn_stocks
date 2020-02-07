# LSTM Stock Price Analysis
Basic utilization of LSTM for predicting stock price

##### Modules used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
```

##### Read in data
```python
prices = pd.read_csv('data/AAPL_5yr.csv')
prices.head()
```
