########## ML - EINFACHES MODELL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


#### Daten laden & vorverarbeiten
# Format:
#    Jede Reihe eine Obeservation, jede Spalte ein Feature
# =>  Shape: X.shape = (no_observations,no_features), y.shape = (no_observations,)
# Scikit-learn: X = Feature Array, y = Response Variable
boston = pd.read_csv('~/Documents/Data/Boston.csv')
boston.columns = [c.upper() for c in boston.columns]
X = boston.drop('MEDV',axis=1)
y = boston['MEDV']


#### Model trainieren
reg = LinearRegression()
reg.fit(X,y)

#### Vorhersage
y_pred = reg.predict(X)

#### Koeffizienten
pd.DataFrame(reg.coef_,X.columns)
reg.intercept_



#### Evaluation mittels Train/Test-Split
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
reg.fit(X_train,y_train)

## Metriken berechnen & ausgeben
r2     = reg.score(X_test, y_test)
y_pred = reg.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
print("R^2: {}".format(r2))
print("Root Mean Squared Error: {}".format(rmse))


### Cross-Validation
reg = LinearRegression()
cv_scores = cross_val_score(reg,X_test,y_test,cv=10)
plt.boxplot(cv_scores);
print(cv_scores) # r^2 werte
print(np.median(cv_scores)) # statistiken berechnen
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
