import ee
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Earth Engine
ee.Authenticate(force=True)
ee.Initialize()

# 2. Region and time
india = ee.Geometry.Rectangle([68.0, 8.0, 97.0, 37.0])
start = '2024-01-01'
end   = '2024-12-31'

# 3. MODIS datasets and compute indices
ndvi = ee.ImageCollection("MODIS/061/MOD13Q1") \
    .filterBounds(india) \
    .filterDate(start, end) \
    .select("NDVI") \
    .mean()

lst = ee.ImageCollection("MODIS/061/MOD11A2") \
    .filterBounds(india) \
    .filterDate(start, end) \
    .select("LST_Day_1km") \
    .mean()

mod09 = ee.ImageCollection("MODIS/061/MOD09GA") \
    .filterBounds(india) \
    .filterDate(start, end) \
    .median() \
    .select(['sur_refl_b02', 'sur_refl_b06'], ['NIR', 'SWIR'])

ndmi = mod09.normalizedDifference(['NIR', 'SWIR']).rename("NDMI")

landcover = ee.ImageCollection("MODIS/061/MCD12Q1") \
    .filterBounds(india) \
    .filterDate(start, end) \
    .first() \
    .select("LC_Type1")

forest = landcover.eq(1).rename("Forest")

burn = ee.ImageCollection("MODIS/061/MCD64A1") \
    .filterBounds(india) \
    .filterDate(start, end) \
    .select("BurnDate") \
    .max() \
    .gt(0) \
    .rename("Fire")

# Combine all bands
img = ndvi.addBands(lst).addBands(ndmi).addBands(forest).addBands(burn)

# Sample points for ML
samples = img.sample(
    region=india,
    scale=1000,
    numPixels=6000,
    seed=42,
    geometries=True  # Keep geometry for later
)

# Conversion to DataFrame with coordinates
data = samples.getInfo()["features"]
rows = []
for f in data:
    prop = f["properties"]
    lon, lat = f["geometry"]["coordinates"]
    prop["longitude"] = lon
    prop["latitude"] = lat
    rows.append(prop)

df = pd.DataFrame(rows).dropna()

# Features and target
X = df[['NDVI', 'LST_Day_1km', 'NDMI', 'Forest']]
y = df['Fire'].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Predicting all points and save CSV
df['predicted_risk'] = model.predict(X)
df.to_csv("fire_risk_predictions.csv", index=False)
print("Predictions saved to fire_risk_predictions.csv")
