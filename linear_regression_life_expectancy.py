import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("all_countries.csv")

#print(df.head())

#print(df.isnull().sum().sort_values(ascending=False).to_string())

df_null_count = df.isnull().sum().sort_values(ascending=False)
good_cols = list(set(df_null_count[df_null_count <= 3].index) &
                 set(df.describe().columns))

df_cleaned = df.dropna(axis="index", how="any", subset=good_cols).copy()

#print(df_cleaned.isnull().sum().sort_values(ascending=False).to_string())

model = LinearRegression(fit_intercept=True)
X = df_cleaned[[x for x in good_cols if x != "life_expectancy"]]
Y = df_cleaned["life_expectancy"]
model.fit(X, Y)
print(model.coef_)
print(model.intercept_)
print(model.score(X, Y))

'''
for col, coef in zip(X.columns, model.coef_):
    print("%s: %.3e" % (col, coef))
'''

# Selecciona las 5 características más importantes
from sklearn.feature_selection import SequentialFeatureSelector

# Initializing the Linear Regression model
model = LinearRegression(fit_intercept=True)

# Perform Sequential Feature Selector
sfs = SequentialFeatureSelector(model, n_features_to_select=5)
X = df_cleaned[[x for x in good_cols if x != "life_expectancy"]]
Y = df_cleaned["life_expectancy"]
sfs.fit(X, Y)           # Uses a default of cv=5
selected_feature = list(X.columns[sfs.get_support()])
print("Feature selected for highest predictability:", selected_feature)

# Usamos las características importantes
model = LinearRegression(fit_intercept=True)
X = df_cleaned[selected_feature]
Y = df_cleaned["life_expectancy"]
model.fit(X, Y)
print(model.score(X, Y))
for col, coef in zip(X.columns, model.coef_):
    print("%s: %.3e" % (col, coef))
print("Intercept:", model.intercept_)