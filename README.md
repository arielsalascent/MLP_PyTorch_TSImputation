# Imputation model for multivariate time series using PyTorch

## Problem description:
The long periods with missing data among the series obstaculize the perfomance of deep time series analysis. At the same time de prediction task with an incomplete time series could became unreliable.

- The data is for this project came from the repository of the [University of Wyoming](https://weather.uwyo.edu/upperair/sounding.html)
- The imputation process it is based on the [Part ek al. (2022)](https://www.researchgate.net/publication/366552360_Long-term_missing_value_imputation_for_time_series_data_using_deep_neural_networks)
- The data has sporadic and long gaps with missing data.

### Import the libraries needed and connect to the Google Colab GPU

```python
#pip install -r requirements.txt
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
!pip install -q install torchmetrics
import torchmetrics
from sklearn import metrics
from tqdm.notebook import tqdm, trange
device = 'cuda' if torch.cuda.is_available() else 'cpu'
!nvidia-smi
```
### Import the CSV

```python
file_name='DB_AP_L.csv'
folder_path = os.path.join(os.getcwd())
datos_AP = pd.read_csv(os.path.join(folder_path, file_name), sep=';')
datos_AP.index = datos_AP["Fecha"]
datos_AP = datos_AP.drop(columns=["Fecha"])
print(datos_AP.head())
```






