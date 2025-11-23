# Modelling and Predicting Crop Health Using Remote Sensing Data and Neural Networks

Welcome to my project! 🌱📡🚀  
Here, we explore **remote sensing data** to **assess vegetation health** and **predict crop health** using a **Neural Network** built with TensorFlow/Keras.  

---

## 📌 Project Overview

This project focuses on:

- Processing **raster and vector data** (DEM, DTM, Orthophoto, and shapefiles).  
- Calculating **NDVI (Normalized Difference Vegetation Index)** to measure vegetation health.  
- Building a **Neural Network** classifier to distinguish **healthy vs unhealthy crops**.  
- Evaluating the model using **accuracy, precision, recall, ROC AUC**, and a **confusion matrix**.  

---

## 🛠️ Libraries & Tools Used

```bash
# Python libraries
numpy
pandas
rasterio
geopandas
matplotlib
seaborn
sklearn
tensorflow
rasterstats
```

##🌾 Data

The dataset includes:

dem.tif → Digital Elevation Model

dtm.tif → Digital Terrain Model

ortho.tif → Orthophoto (multiband)

plots_1.shp & plots_2.shp → Plot boundaries

Data was originally sourced from DroneMapper's GitHub.

##📊 Feature Engineering

1. Mask invalid values

Elevation < 0 → NaN
Thermal values ≤ 0 → NaN and converted to °C

2. Compute NDVI

```json
ndvi = (NIR - Red) / (NIR + Red)
ndvi = np.where(np.isnan(ndvi), 0, ndvi)
```

3. Zonal statistics per plot

Mean NDVI
Mean Elevation
Mean Thermal
Mean DTM

4. Synthetic Target Variable

Healthy crops (NDVI between 0.4–0.8) → 1
Others → 0

##⚖️ Handling Class Imbalance

The dataset was slightly imbalanced:
0 (unhealthy crops) : 78
1 (healthy crops)   : 54

undersampled the majority class to create a balanced dataset.

##🧠 Neural Network Model

Architecture:
```json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation="relu", input_shape=(x_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])
```

##🏋️ Training the Model

Epochs: 100
Batch size: 32
Validation split: 20%
Model achieved strong performance on the validation set.

##📈 Evaluation Metrics
```json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
```

Accuracy: from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

Accuracy: 0.9090909090909091
Precision: 1.0
Recall: 0.8333333333333334
ROC AUC Score: 1.0

##Confusion Matrix
```json
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

The model accurately distinguishes healthy from unhealthy crops, with over 90% accuracy.








