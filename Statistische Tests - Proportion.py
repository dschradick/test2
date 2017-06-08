########## STATISTISCHE TESTS - PROPORTION
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
import math
import matplotlib.pyplot as plt
import seaborn as sns

### Vorbereitungen
np.random.seed(0)

#### Sample-Größe bestimmen
# Von 1.5% zu 2.5% => 1%
# bei Signifikanzlevel von 5% und 80% Power
import statsmodels.stats.api as sms

es = sms.proportion_effectsize(0.015, 0.025)
sms.NormalIndPower().solve_power(es, power=0.8, alpha=0.05, ratio=1)

### One-sample z-test
# count = Anzahl der Erfolge in nobs trials
# nobs  = Anzahl der trials / observations
# value = wert in H0
# hier: proportion=15/100
count = 15
nobs = 100
value = .05
stat, pval = proportions_ztest(count, nobs, value, "larger")
print('{0:0.3f}'.format(pval))

### Two-Sample z-test
# Proportions: 5/83 und 12/99
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

count = np.array([5, 12])
nobs = np.array([83, 99])
stat, pval = proportions_ztest(count, nobs)
stat
print('{0:0.3f}'.format(pval))

## Ausgabe wie in R
from scipy.stats import chi2_contingency
import numpy as np

proportions = np.array([[5, 12],
                        [83, 99]])  # contingency table
# => beinhaltet die beobachteten frequencies für die jeweiligen Kategorien

result = chi2_contingency(observed=proportions)
print('chi2-stat: {0:0.3f}'.format(result[0]))
print('p-value: {0:0.3f}'.format(result[1]))
print('expected frequencies:\n {}'.format(result[3]))
# => basierend auf den marginal sums der tabelle

