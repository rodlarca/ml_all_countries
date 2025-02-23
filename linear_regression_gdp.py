import pandas as pd

# Load data
df = pd.read_csv("all_countries.csv")

print(df.head())

df_null_count = df.isnull().sum().sort_values(ascending=False)
good_cols = list(set(df_null_count[df_null_count <= 3].index) &
                 set(df.describe().columns))
print(good_cols)

df_cleaned = df.dropna(axis="index", how="any", subset=good_cols).copy()
print(df_cleaned)

from sklearn.linear_model import LinearRegression

print("\nVariable objetivo: gdp, características: polulation -----------------------------------------------------------------------------------------")
model = LinearRegression(fit_intercept=True)
X = df_cleaned[["population"]]
Y = df_cleaned["gdp"]
model.fit(X, Y)
print(model.coef_)
print(model.intercept_)
print(model.score(X, Y))

print("\nVariable objetivo: gdp, características: polulation, rural_population, median_age, life_expectancy ------------------------------------------")
model = LinearRegression(fit_intercept=True)
X = df_cleaned[["population", "rural_population", "median_age", "life_expectancy"]]
Y = df_cleaned["gdp"]
model.fit(X, Y)
print(model.coef_)
print(model.intercept_)
print(model.score(X, Y))