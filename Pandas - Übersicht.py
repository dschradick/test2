 ########## PANDAS - ÜBERSICHT
#
# !!! McKinney - Python for Data Analysis!!!
#
from ggplot import mtcars, diamonds
%cd ~
#### IMMER: DIESE IMPORTS UND SETTINGS
#-----------
import warnings
warnings.simplefilter(action='ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pd.options.display.max_rows = 10
plt.style.use('ggplot')
plt.rc("axes.spines", top=False, right=False)  # despine() als default
#-----------

## !! to 'object' !!!
mtcars['cyl'] = mtcars.cyl.astype(str)

# Aggregation & grouping
## Effektstärke
from scipy import stats
np.random.seed(seed=0)
effect = 2
A = norm.rvs(size=1000,loc=2,scale=10)
B = norm.rvs(size=1000,loc=2,scale=10)
ttest_ind(A,B+effect)
group1 = [[u'desktop', 14452], [u'mobile', 4073], [u'tablet', 4287]]
group2 = [[u'desktop', 30864], [u'mobile', 11439], [u'tablet', 9887]]
obs = np.array([[14452, 4073, 4287], [30864, 11439, 9887]])
chi2, p, dof, expected = stats.chi2_contingency(obs)
print(p)

#### EINLESEN
path = '~/Documents/Data/german_credit_test.csv'
open(path).readline() # => struktur angucken
csv_filepath = '~/Documents/Data/german_credit_test.csv'
col_names = ['col_1','col_2','col_3']
csv_data = pd.read_csv(csv_filepath, index_col=0, names=None, na_values = [0,-1], sep=',',nrows=None)
xls_filepath = 'Documents/Data/SuperstoreSample.xls'
xls_data = pd.read_excel(xls_filepath,'Orders')


#### SEPERATING NUMERICAL AND CATEGORIAL
custid = ['customerID']; target = ['Churn']

categorical = data.nunique()[data.nunique()<10].keys().tolist()
categorical.remove(target[0])
numerical = [col for col in data.columns
             if col not in custid+target+categorical]

### DUMMIES ERZEUGEN
categorials = ['Type of apartment','Occupation','Foreign Worker','Sex & Marital Status','Purpose']
X[categorials] = X[categorials].astype('category')
X = pd.get_dummies(X, drop_first=True) # => konvertiert alle object & categorials - andere bleiben wie sie sind


#### CHAINING
(mtcars
    .query('hp > 100')
    .groupby('cyl')
    .mean())


#### DIVERSES
## Auswählen von Spalten basierend auf Typ
mtcars.select_dtypes(include=['number']);              # 'object','datetime'
# Spaltennamen setzen
names = list(mtcars.columns); mtcars.columns = names
## Index aus Spalte generien
mtcars.set_index("name", drop=False, inplace=True)
## Spaltennamen bereinigen
mtcars.columns = mtcars.columns.str.replace(' ', '-')  # Kein "_" => sonst Probleme mit seaborn
mtcars.columns = list(map(str.lower, mtcars.columns))  # Spaltennamen zu lower-case
# Flattening von hierarchischen Spalten-index (z.B. nach mehrfach-aggregagtion)
df.columns = ['_'.join(col).strip() for col in df.columns.values]
df.columns = [col[1].strip() for col in df.columns.values]  # oberes level entfernen
df.columns =  df.columns.droplevel(level=0)



# MAP
a = [1,2,3,4,5]
list(map(lambda x: x*2,a))

#### VISUALSIEREN

# Titel + Achsenbeschriftung
ax = (diamonds.carat.plot(
    kind='hist',
    figsize=(8,6),
    xticks=range(1,4),
    title='TITEL'.upper()))
ax.set_xlabel("x label (unit)");
ax.set_ylabel("y label (unit)")
#---
plt.figure(figsize=(10,10))
sns.heatmap(mtcars.corr(),annot=True,cmap='Blues')    # Korrelationsmatrix
mtcars.hist(bins=10, figsize=(10,10))                 # Histomgram für alle Spaltennamen
mtcars.plot.scatter(x='disp',y='hp',c='qsec')         # Scatterplot mit colorbar
mtcars.groupby('cyl').hist()                          # Histogram für alle Spalten nach Gruppe
mtcars.groupby('cyl').hp.hist(alpha=0.4)              # Überlagertes(!) Histogram für Gruppen
mtcars.groupby('cyl')['hp'].mean().plot(kind='barh')  # !!!! Mittelwerte von Gruppen bzgl. Variable
mtcars[['hp','qsec']]\
    .plot(kind='box', subplots=True, layout=(1,2))    # Boxplots von mehreren Spalten
# Zählen von Werten und Kategorien
mtcars.nunique().plot(kind='barh')                    # Im DF: Anzahl verschiedene distinct Werte
mtcars.cyl.nunique()                                  # Wieviele levels = distinct Werte
mtcars.cyl.value_counts()                             # Anzahl der Vorkommnisse von jedem distinct wert
mtcars.cyl.value_counts().plot(kind='barh')           # In Spalte: Häufigkeit der einzelnen distinct Werte
# Area => auf Reihenfolge achten - erste ist x!
mtcars.groupby(['cyl','am']).size().unstack().plot(kind='area',stacked=True)
# Legende nach rechts aussen
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

## 3 kontinuierliche Variablen
df = (diamonds
      .groupby((pd.cut(diamonds.depth,20),  # Y
                pd.cut(diamonds.carat,20))  # X
              )['price']                    # Ziel Variable
      .max() # Statistik: count, sum, mean, median, max, min,...
              # => rumspielen
      .unstack()
      .sort_index(ascending=False))
# Heatmap: 'coolwarm' => von minus bis plus (z.B. Korrelation)
#          'Blues' => wenn nur aufsteigend (z.B. counts)
sns.heatmap(df, annot=False, fmt=".0f", linewidths=.05, cmap='Blues');


### Contingency-tables
table = pd.crosstab(index=mtcars.am,
                    columns=[mtcars.cyl,mtcars.gear],
                    margins=True)
table          # crosstable
table[4][3]    # Spalte

## Untergruppen(Anzahl): Aggregation bzgl 2 Variablen
pd.crosstab(mtcars.cyl,mtcars.carb).plot(kind='barh')
# Beliebige Aggregation 2er Variablen => eine Targetvariable
# 1. Barchart oder 2. Heatmap
# => gut bei ordinalen Variablen
# => visualisieren von Interaktionseffekten
fig = plt.figure(figsize=(12,5))
ax = (sns.heatmap(                                  # Aggregation bzgl 2 Variablen
        mtcars.groupby(['cyl','carb'])
        .hp.mean()
        .unstack(),
        linewidths=.5, fmt=".1f",cmap='Blues',
        annot=True, cbar=True))
ax.set_xlabel("carb"); ax.set_ylabel("cyl")



#### DATEN BESCHREIBEN
mtcars.head(5).transpose()                     # => gut wenn viele Spalten!
mtcars.describe().transpose()
mtcars.shape; mtcars.dtypes, mtcars.nunique()
mtcars.info(); mtcars.describe()               # Allgemein
## Distinkte Werte und Werte zählen
mtcars.nunique() # auf df!!                    # Anz. der (distinct) Werte in JEDER SPALTE
# - Kategorische
mtcars.cyl.unique()                            # Welche distinkten Werte gibt es
mtcars.cyl.value_counts()                      # ANZ. der WERTE IN LEVEL(!) in Kategorien
# -
mtcars.groupby('cyl').size.sort_values()       # =...
freq_table = pd.crosstab(mtcars.cyl,mtcars.carb)
sns.heatmap(freq_table,annot=True, fmt="d", linewidths=.05, cmap='Blues')
# oder
 # => beide mit plot(kind='barh')
sns.heatmap(mtcars.corr(),annot=True)          # Korrelationsmatrix
mtcars.cyl.max(); mtcars.cyl.idxmax()          # Max-Element und Index
mtcars.quantile([0.25,0.75])                   # Quantile


### Sortieren
mtcars.sort_values('hp',ascending=True)
mtcars.sort_values(['hp', 'mpg'], ascending=True, inplace=False)


#### BEOBACHTUNGEN
### Zuweisung zu einem Subset
mtcars.loc[mtcars.hp < 100,'hp'] = 100         # wird inplace gemacht
### AUSWAHL
mtcars.head(1); mtcars.tail(1)                 # Anfang / Ende von df
mtcars.query('carb > 4')                       # Filter
# => nicht performant - besser: loc
mtcars.iloc[1:10]                              # Position
mtcars.loc['Mazda RX4':'Datsun 710']           # Reihenname
mtcars[mtcars.carb > 4]                        # =...
mtcars[mtcars.carb > 4][['cyl','mpg']]         # Filter + Spaltenauswahl
mtcars.loc[mtcars.carb > 4,['cyl','mpg']]      # =...
mtcars.query('carb > 4 & hp > 150')            # Fiter mehrerer Variablen
mtcars[(mtcars.carb > 4) & (mtcars.hp > 150)]  # =...
mtcars[mtcars.name.str.startswith('Maz')]
mtcars[mtcars.name.str.contains('Maz')]        # Filter nach String
mtcars[~mtcars.name.str.contains('Maz')]       # Filter nach nicht(!) spezieller String
np.where(mtcars.mpg > 20)                      # Indizes, welche Bed. erfüllen
mtcars.cyl.drop_duplicates()                   # distinct
mtcars.cyl.unique()                            # distinct für kategorisch
mtcars.cyl.sample(n=10)                        # sample
mtcars.cyl.sample(frac=0.1)                    # sample


#### SPALTEN
### Auswahl
## Einfach
mtcars.hp                                      # ein Spalte nach Name
mtcars['hp']                                   # =...
mtcars[['hp', 'cyl']]                          # Mehrere Spalten
mtcars[mtcars.columns[1]]                      # Spaltennummer
## Komplexere Auswahl
selection = mtcars.columns.str.contains('cyl')
mtcars[mtcars.columns[selection]]              # Filter auf Namen - Alternativ..
mtcars.filter(regex='^cy',axis=1)              # Regex auf Spaltenname
mtcars.selectdrop(['cyl','carb','disp'],axis=1)# ALLE SPALTEN AUSSER
mtcars.select_dtypes(include=['object'])       # 'number','object','datetime'
### Löschen
del mtcars['name']                             # Spalte löschen
### Hinzufügen
# NP-where zum erzeugen neuer Spalten
mtcars['car_type'] = np.where(mtcars.mpg > 20,'suv','smart')  #  if,then,else
mtcars['new_col'] = range(0,32)                # Werte
mtcars['new_col'] = mtcars.hp - mtcars.mpg     # auf Basis anderer Spalten
mtcars.assign(new_col=mtcars.hp - mtcars.mpg)  # =... => für chaining
 # => assign für chaining (wie mutate)
### Umbenennen
mtcars.columns                                 # ist index (hat mit .str funktionen)
mtcars.rename(columns={'hp': 'ps'})            # Umbenennen



#### REIHEN
mtcars.head(1)
mtcars['name'] = mtcars.index                  # Index => Spalte
mtcars.set_index('name')                       # Spalte => index
mtcars = mtcars.reset_index(drop=True)         # Erzeugt normalen numerischen Index #  # (drop = True =>  alter index nicht als Spalte anlegen)
mtcars.index = mtcars.name                     # alternativ



#### Reihe + Spalte
# [Reihe,Spalte]
mtcars.iloc[1:10,:]                                            # Numerisch: [Reihe,Spalte]
mtcars.iloc[:,1:10]
mtcars.iloc[1:10,1:3]
mtcars.loc['Mazda RX4':'Datsun 710',['hp', 'cyl']]             # Nach Namen
# mtcars.iloc[0:10,'hp'] geht nicht # => kein mixing
## Mischen von integer und string adressierung
mtcars.ix[0:10,'hp'] # => geht, aber deprecated                # Numerisch + Namen mischen => deprecated
# Statt .ix - besser:
mtcars.iloc[0:10, mtcars.columns.get_loc('hp')]                # ix für eine column
mtcars.iloc[0:10, mtcars.columns.get_indexer(['hp', 'cyl'])]   # ix für mehrere columns
mtcars.loc[mtcars.index[0:10], 'hp']                           # ...alternativ




#### AGGREGATION
# => entweder auf ganzen Dataframe / Spalte / Subset
mtcars.describe()                              # Dataframe
mtcars.mpg.describe()                          # Spalte
mtcars.query('mpg > 20').describe()            # Subset
## Anzahl & Proportion
mtcars.cyl.count()                             # Anzahl der Einträge
mtcars.cyl.value_counts()                      # Anzahl Einträge nach Faktor
np.sum(mtcars.hp > 150)                        # Anzahl mit Bedingung
np.mean(mtcars.hp > 150)                       # Proportion
## Gruppierung
mtcars.groupby('cyl').mean()                   # auf ganzen dataframe berechnen
mtcars.groupby('cyl')['hp'].count()            # nur auf spalte
mtcars.groupby('cyl').agg({'hp': 'mean'})      # = ...
mtcars.groupby('cyl')['hp'].count().plot(kind='bar')
## HAVING
# (nicht spalte nach groupby angeben(!!!)
# having count(*) > 12
(mtcars
    .groupby('cyl')
    .filter(lambda x: len(x) > 12)
# select sum(mpg) from mtcars where am = 1 group by cyl having sum(mpg) > 100
(mtcars
    .query('am == 0')                         # WHERE
    .groupby('cyl')                           # GROUP BY für HAVING
    .filter(lambda x: x['mpg'].sum() > 100)   # HAVING-Bed
       # => liefert subset von originalen df mit den original Werten!! (nicht gruppiert)
    .groupby('cyl')['mpg']                    # Eigentliches GROUP BY
    .sum())                                   # Aggregation für Gruppe
## Komplexe Grupperierung / Aggregationen
# Eine Spalte mit mehreren Funktionen
mtcars.groupby('cyl').agg({'hp':[min,max,'mean']})  # mehrere funktionen
##  Mehrere Spalten mit mehreren Funktionen
aggregations = {
    'hp' : [min,max,'mean'],
    'mpg': [min,max,'mean']}
df = mtcars.groupby('cyl').agg(aggregations); df
# => nach aggregation multiindex in spalten entfernen
df.columns = ['_'.join(col).strip() for col in df.columns.values]; df
## Umbenennen
# z.B. hp-unterlevel-min  ==> hp_min
grouped = mtcars.groupby('cyl').agg({'hp':[min,max,'mean']}); grouped
grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]; grouped
## Transform -
# fügt werd in die ensprechenden Werte in Zellen einer kopie des original-df ein
mtcars.groupby('cyl')['hp'].transform('mean')
## Levels - Gruppen nach Level im MultiIndex
mtcars2 = mtcars.set_index(['cyl'], append=True)    # Hinzufügen von cyl zu index
mtcars2.groupby(level=1).sum()


#### WINDOW FUNCTIONS
# SELECT cyl,hp,sum(hp) OVER (PARTITION BY cyl ORDER BY hp )
mtcars['hp_cumsum'] = (mtcars.sort_values('hp')
                             .groupby('cyl')
                             ['hp']
                             .cumsum());   #  alternativ: .transform(np.cumsum)); .transform(lambda x: x.rank()
mtcars[['cyl','hp','hp_cumsum']].sort_values(['cyl','hp_cumsum'])
# SELECT cyl,hp,sum(hp) OVER (PARTITION BY cyl ORDER BY hp rows between 2 preceding and current)
mtcars['hp_cumsum'] = (mtcars.sort_values('hp')
                             .groupby('cyl')
                             ['hp']
                             .apply(lambda x : x.rolling(3).sum()));
mtcars[['cyl','hp','hp_cumsum']].sort_values(['cyl','hp_cumsum']).head(5)
# select cyl,hp,rank() over (partition by cyl order by hp desc)
mtcars['hp_rank'] = (mtcars.sort_values('hp')
                             .groupby('cyl')['hp']
                             .rank(method='first', ascending=False));
mtcars[['cyl','hp','hp_rank']].sort_values(['cyl','hp_rank']).head(20)


#### CLEANING
# Strings aufräumen
diamonds.cut.str.replace('Fair','Ok')              # STRINGFUNKTIONEN
pattern = '|'.join(['Fair', 'Ok', 'Normal'])
diamonds.cut.str.replace(pattern, 'OK', regex=True)
replacement = {
    "Fair": "OK",
    "Ok": "OK",
    "HELLO": "CORP"
}
diamonds.cut.str.replace(replacement, regex=True)
# Einträge löschen - (generelles slicen)
mask = diamonds.cut.str.contains('Good')
mask = ~diamonds.cut.str.endswith('Good')
diamonds[mask]
# String-column aufsplitten
new = diamonds.cut.str.split(' ',n = 1,expand = True)
diamonds["First"] = new[0]
diamonds["Second"] = new[1]
### Type-Casting
mtcars['cyl'] = mtcars.cyl.astype(str)
# Wenn voher object => pd.to_numeric
mtcars['cyl'] = pd.to_numeric(mtcars.cyl, errors='coerce')
# Wenn voher schon number => z.B.  int8|16|32|64,float64,boolean
mtcars['cyl'] = mtcars.cyl.astype('int8')
### Duplikate
mtcars.drop_duplicates()                           # Duplikate entfernen
mtcars.drop_duplicates(subset='cyl', keep='first') # Duplikate entfernen (wenn duplikat auf id)
### Missing Values
mtcars.isnull().any().any()                        # Gibt es nans in df?
mtcars.isnull().any()                              # In welcher Spalte gibt es nans?
mtcars.isnull().sum().sum()                        # Wieviele NANs in df?
mtcars.isnull().sum()  #!!!                        # Wieviele NANs in welcher Spalte?
## Zeig die Reihen
data.loc[mtcars.T.isnull().any()]                  # Alle Reihen mit NANs
data.loc[mtcars.cyl.isnull()]                    # Reihen mit NANs in Carat
## Nans behandeln
mtcars.dropna(axis=0, how='any')                   # Droppen: wo "any" or "all" der Daten Missing
mtcars.dropna(subset=['name', 'cyl'])              # Droppen: wenn spezielle Spalten missing haben
mtcars.cyl.fillna(mtcars.cyl.mean())               # Ersetzen von Nan
mtcars.cyl.replace(to_replace=4, value=10)         # Ersetzen: von 4 durch 10
mtcars.loc[mtcars.hp < 100,'hp'] = 100             # ganzes subset




#### FREQUENZ-TABELLE
ft = pd.value_counts(mtcars.cyl);ft                                   # Frequency-Table
ft / ft.sum()                                                         # Frequency-Table Proportionen
pd.crosstab(mtcars.cyl,mtcars.carb,margins=True)                      # Anzahl
pd.crosstab(mtcars.cyl,mtcars.carb,margins=True, normalize='index')   # Proportion
pd.crosstab(mtcars.cyl,mtcars.carb,margins=True).plot(kind='barh')



#### DISKRETISIEREN
## Binäre Variable erzeugen
# 1. Vektor-Version
mtcars['binary_variable'] = mtcars.hp > 5
# 2. Reihenweise
## A. Einfach
mtcars['binary_category'] = mtcars.hp.apply(lambda x: 0 if x <= 120 else 1)
## B. Funktion
def discretize_target(x):
    if x < 70:
        return "low"
    elif x < 100:
        return "med"
    else:
        return "high"
mtcars['mpg_cat'] = mtcars.hp.apply(discretize_target)
## Range zerlegen
bins = np.arange(0,5,0.5)         # range nimmt keine floats!
labels=['low', 'medium', 'big']                                  # n labels
bins = [mtcars.hp.min()-1, 100, 200, mtcars.hp.max()+1]              # dafür n+1 grenzen
mtcars['hp_cat'] = pd.cut(mtcars.hp, bins=bins, labels=labels)   # kategorisch erstellen
# Komplex: über function => siehe titanic


### APPLY & MAP
square = lambda x: x ** 2
mtcars['squared_hp'] = mtcars.hp.apply(square)
# =>  standard python: list comprehension ist syntactic sugar für map/apply
mtcars['squared_hp'] = list(map(square,mtcars.hp))

#### Reordering von Factors
mtcars['cyl'] = mtcars['cyl'].astype('category')
mtcars['cyl'] = mtcars['cyl'].astype(str)
mtcars['cyl'] = mtcars['cyl'].cat.reorder_categories([4, 6,8])



#### DATE-TIME
### Konvertieren
mtcars['build_date'] = "2017-01-01 12:10:02"              # Datetime String
mtcars['build_date'] = pd.to_datetime(mtcars.build_date)  # Konvertierung
pd.to_datetime("2017-1-1-15",infer_datetime_format=True)
pd.to_datetime('20141101 1211', format='%Y%m%d %H%M', errors='ignore')
### Extrahieren
mtcars.build_date.dt.year
mtcars.build_date.dt.minute
mtcars.build_date.dt.strftime('%m/%d/%Y %H:%M')
# %Y Four-digit year %y Two-digit year
# %m Two-digit month [01, 12] %d Two-digit day [01, 31]
# %H Hour (24-hour clock) [00, 23]
# %I Hour (12-hour clock) [01, 12]
# %M Two-digit minute [00, 59]
# %S Second [00, 61] (seconds 60, 61 account for leap seconds) %w Weekday as integer [0 (Sunday), 6]
# %U Week number of the year [00, 53]; Sunday is considered the first day of the week, and days before the first Sunday of the year are “week 0”
# %W Week number of the year [00, 53]; Monday is considered the first day of the week, and days before the first Monday of the year are “week 0”
# %z UTC time zone offset as+HHMMor-HHMM; empty if time zone naive %F Shortcut for%Y-%m-%d(e.g.,2012-4-18)
# %D Shortcut for%m/%d/%y(e.g.,04/18/12)


### Gruppieren
mtcars['build_date'] = pd.date_range(start='2017-01-01', periods=len(mtcars))
mtcars.groupby('build_date').count().plot()
mtcars.groupby(mtcars.build_date.dt.year)['mpg'].count().plot()
### Auswahl
mtcars[mtcars.build_date > '2017-01']
mtcars[mtcars.build_date > '2017-01-30']
