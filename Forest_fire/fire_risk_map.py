import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np

# 1. Loading predictions
df = pd.read_csv("fire_risk_predictions.csv")

# 2. Conversion to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
gdf = gdf.to_crs(epsg=3857)  # For basemap

# 3. continuous heatmap
fig, ax = plt.subplots(figsize=(12,12))

hb = ax.hexbin(
    gdf.geometry.x,
    gdf.geometry.y,
    C=gdf['predicted_risk'],
    gridsize=100,
    reduce_C_function=np.mean,
    cmap='hot',
    mincnt=1
)

# Colorbar
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Predicted Fire Risk')

#  basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# Titles
plt.title("Continuous Forest Fire Risk Heatmap - India", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig("india_fire_risk_heatmap_continuous.png", dpi=300)
plt.show()
