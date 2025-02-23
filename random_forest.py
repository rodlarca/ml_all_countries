import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("all_countries.csv")

df_null_count = df.isnull().sum().sort_values(ascending=False)
good_cols = list(set(df_null_count[df_null_count <= 3].index) &
                 set(df.describe().columns))

df_cleaned = df.dropna(axis="index", how="any", subset=good_cols).copy()

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

df_cleaned["north"] = df_cleaned["latitude"] > 0

model = DecisionTreeClassifier(max_depth=3)
X = df_cleaned[col_to_use]
Y = df_cleaned["north"]
model.fit(X, Y)
print(model.score(X, Y))