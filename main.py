import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("all_countries.csv")

# Basic exploration of a dataset
print(df.info())

print(df.head())

print(df.tail())

print(df.sample(5))

# Filling in missing values
#df_filled = df.fillna(0)

print(df.isnull().sum().sort_values(ascending=False).to_string())

# Filter for rows where the count of missing values is zero
df_null_count = df.isnull().sum().sort_values(ascending=False)
print(df_null_count[df_null_count == 0].index)

# Identify numerical columns
print(df.describe().columns)

print(list(set(df.describe().columns) & set(df_null_count[df_null_count == 0].index)))

# Identify the numeric columns that have three or fewer missing values
df_null_count = df.isnull().sum().sort_values(ascending=False)
good_cols = list(set(df_null_count[df_null_count <= 3].index) &
                 set(df.describe().columns))
print(good_cols)

df_cleaned = df.dropna(axis="index", how="any", subset=good_cols).copy()
print(df_cleaned)

# Build a linear model
model = LinearRegression(fit_intercept=True)
X = df_cleaned[["population"]]
Y = df_cleaned["gdp"]
model.fit(X, Y)
print(model.coef_)
print(model.intercept_)
print(model.score(X, Y))

# Adding more columns to X
model = LinearRegression(fit_intercept=True)
X = df_cleaned[["population", "rural_population", "median_age", "life_expectancy"]]
Y = df_cleaned["gdp"]
model.fit(X, Y)
print(model.coef_)
print(model.intercept_)
print(model.score(X, Y))

# Linear regression for life expectancy using all relevant factors
df_null_count = df.isnull().sum().sort_values(ascending=False)
good_cols = list(set(df_null_count[df_null_count <= 3].index) &
                 set(df.describe().columns))
print(good_cols)

X = df_cleaned[[x for x in good_cols if x != "life_expectancy"]]
Y = df_cleaned["life_expectancy"]
model.fit(X, Y)
print(model.coef_)
print(model.intercept_)
print(model.score(X, Y))

for col, coef in zip(X.columns, model.coef_):
    print("%s: %.3e" % (col, coef))

# Identify the top five factors affecting life expectancy.

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

# Build the model again and examine the coefficients:
model = LinearRegression(fit_intercept=True)
X = df_cleaned[selected_feature]
Y = df_cleaned["life_expectancy"]
model.fit(X, Y)
print(model.score(X, Y))
for col, coef in zip(X.columns, model.coef_):
    print("%s: %.3e" % (col, coef))
print("Intercept:", model.intercept_)

# Convert GDP, land area, and some other columns into their “per capita” versions
per_capita = ["gdp", "land_area", "forest_area", "rural_land", "agricultural_land",
              "urban_land", "population_male", "population_female", "urban_population",
              "rural_population"]
for col in per_capita:
    df_cleaned[col] = df_cleaned[col] / df_cleaned["population"]

col_to_use = per_capita + [
    "nitrous_oxide_emissions",  "methane_emissions", "fertility_rate", "hospital_beds",
    "internet_pct", "democracy_score", "co2_emissions", "women_parliament_seats_pct",
    "press", "electricity_access_pct", "renewable_energy_consumption_pct"]

model = LinearRegression(fit_intercept=True)
sfs = SequentialFeatureSelector(model, n_features_to_select=6)
X = df_cleaned[col_to_use]
Y = df_cleaned["life_expectancy"]
sfs.fit(X, Y)           # Uses a default of cv=5
selected_feature = list(X.columns[sfs.get_support()])
print("Feature selected for highest predictability:", selected_feature)

# Decision Tree, explore whether countries in the Northern and Southern Hemispheres differ

df_cleaned["north"] = df_cleaned["latitude"] > 0

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3)
X = df_cleaned[col_to_use]
Y = df_cleaned["north"]
model.fit(X, Y)
print(model.score(X, Y))

print(Y.value_counts())

# Lesson 07: Random Forest and Probability

print("\nLesson 07: Random Forest and Probability ---------------------------------------------")

from sklearn.ensemble import RandomForestClassifier

print("\nRandomForestClassifier ---------------------------------------------------------------")
model = RandomForestClassifier(n_estimators=5, max_depth=3)
X = df_cleaned[col_to_use]
Y = df_cleaned["north"]
model.fit(X, Y)
print(model.score(X, Y))

from sklearn.ensemble import GradientBoostingClassifier

print("\nGradientBoostingClassifier -----------------------------------------------------------")
model = GradientBoostingClassifier(n_estimators=5, max_depth=3)
X = df_cleaned[col_to_use]
Y = df_cleaned["north"]
model.fit(X, Y)
print(model.score(X, Y))
# print(model.predict_proba(X))

print(model.predict(X))

import numpy as np

print(np.mean(model.predict_proba(X)[range(len(X)), model.predict(X).astype(int)]))