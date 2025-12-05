# Forest_Fire_Detection
Forest-fire risk prediction system for Indian terrain using MODIS satellite data. Includes automated Earth Engine extraction, NDVI/LST/NDMI processing, burned-area labeling, ML training with Random Forest, and geospatial heatmap visualization for research and disaster management.

# Features

- Download MODIS datasets (NDVI, LST, NDMI, Forest cover, Burn history) from GEE.
- Train a **Random Forest classifier** to predict forest fire risk.
- Generate **predicted risk CSV** and **geospatial heatmaps**.
- Fully automated workflow using `forest_fire.py`.
- Modular scripts for easy customization and experimentation.

# Repository Structure




# Quick Start

## 1. Setup

1. Create a Google Earth Engine account: (https://signup.earthengine.google.com/)
2. Clone this repository
3.Install dependencies
   pip install -r requirements.txt

Run 'python forest_fire.py'
