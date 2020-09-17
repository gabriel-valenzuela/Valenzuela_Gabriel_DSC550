# Characterists of a Successful Popular Song

## Data Mining Project 

### Objective

For this analysis, I wanted to understand the aspects of songs such as the beats per minute, the length of the song, the amount of lyrics, or loudness that lead to its popularity in our country. Hence, I chose to use a dataset including the top songs from 2010 to 2019 from Spotify. When people are listening to songs, they usually will not consider these aspects when deciding if they like a song or if it is popular. Therefore, I wanted to see which of these hidden aspects have an effect on how popular a song is at the time. 

### Data Source

https://www.kaggle.com/leonardopena/top-spotify-songs-from-20102019-by-year

### Environment

Python was utilized within a jupyter notebook to complete the analysis

### Methodology

#### Libraries

```python
import pandas as pd
import yellowbrick
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.model_selection import FeatureImportances
from sklearn.ensemble import RandomForestClassifier

from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, f_regression, chi2
import warnings
```

#### Graph Analysis

The first step in the analysis was to load the data and perform exploratory data analysis by describing and summarizing the given information.

```python
data = pd.read_csv("C:\\Users\\Gabe\\Documents\\Bellevue University\\Data Mining\\Project\\Data\\top-spotify-songs-from-20102019-by-year\\top10s.csv", encoding = 'ISO-8859-1')

print("The dimension of the table is: ", data.shape)

# Break data into object not including text or song number
data_values = data[["pop", "spch", "bpm", "nrgy", "dnce", "live", "val", "dur", "acous"]]
print("The dimension of the details table is: ", data_values.shape)

print("Describe Data")
print(data.describe())
print("")
print("Summarized Data")
print(data.describe(include=['O']))



```

Once I have described and observed the data, I chose to specify the features of interest and utitilize histograms to understand the distibution of the selected features. 

```python
# set up the figure size
plt.rcParams['figure.figsize'] = (20, 10)

# make subplots
fig, axes = plt.subplots(nrows = 3, ncols = 3)

# Specify the features of interest
num_features = ['pop','spch', 'bpm', 'nrgy', 'dnce', 'live', 'val', 'dur', 'acous']
xaxes = num_features
yaxes = ['Counts','Counts', 'Counts', 'Counts', 'Counts', 'Counts', 'Counts', 'Counts', 'Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
plt.show()

```
![Popular Song Histograms](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC550/blob/master/Project/PopularSongsHist.png)

Person Ranking
```python
plt.rcParams['figure.figsize'] = (15, 7)

# import the package for visulization of the correlation
from yellowbrick.features import Rank2D

# extract the numpy arrays from the data frame
import numpy as np
X = np.asmatrix(data[num_features])

# instantiate the visualizer with the Covariance ranking algorithm
visualizer = Rank2D(features=num_features, algorithm='pearson')
visualizer.fit(X)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof() # Draw/show/poof the data
plt.show()

```

#### Feature Reduction and Dimensionality



Log-transformation
```python
# log-transformation method using numpy
def log_transformation(data):
    return data.apply(np.log1p)

# check the data for accoustics since it is very skewed
data['Acous_log1p'] = log_transformation(data['acous'])
print(data.describe())

```

#### Model Evaluation and Selection

Feature Selection with Univariate Statistical Tests
```python
X_reg = data_values.iloc[:,1:9]
Y_reg = data_values.iloc[:,0]
# feature extraction
regression_new = SelectKBest(score_func=f_regression, k=4).fit_transform(X_reg, Y_reg)
print(regression_new[:5])
print(X_reg.head())
print(Y_reg.head())

```

Feature Importance using Random Forest Classifier
<br>
![Popular Song Features](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC550/blob/master/Project/PopularSongFeatures.png)

Train and Test for Random Forest Classifier Model

![Confusion Matrix](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC550/blob/master/Project/PopularSongConfusionMatrix.png)
![Classificaion Report](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC550/blob/master/Project/PopularSongClassificationReport.png)
![ROC Rates](https://github.com/gabriel-valenzuela/Valenzuela_Gabriel_DSC550/blob/master/Project/PopularSongRates.png)
