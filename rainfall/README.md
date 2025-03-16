# :umbrella: Predict Rainfall

Source: [Binary Prediction with a Rainfall Dataset](https://www.kaggle.com/datasets/subho117/rainfall-prediction-using-machine-learning)

Columns:
- id: Index
- day: day of the year
- pressure:
- maxtemp:
- temperature:
- mintemp:
- dewpoint:
- humidity:
- cloud:
- sunshine:

## :book: Requisites / Libraries

- Python 3.12: tensorflow not compatible with python 3.13 (11/03/25)
- pandas
- numpy
- scipy
- matplotlib
- seaborn
- plotly
- sklearn
- joblib
- tensorflow

<hr>

## :chart_with_upwards_trend: Analysis

- ### Density

<div>
<img src="readme/density.png" height="600" alt="Density Each">
<img src="readme/density_each.png" height="600">
</div>

- ### Correlation

<img src="readme/correlation.png" width="700">

- ### Matrix Scatter Plot

<img src="readme/scatterplot.png" width="700" height="700">

- ### BoxPlot

<div>
<img src="readme/box_pressure.png" width="400"
 height= "300">
<img src="readme/box_maxtemp.png" width="400" height="300">
</div>
<div>
<img src="readme/box_temperature.png" width="400"
 height= "300">
<img src="readme/box_mintemp.png" width="400" height="300">
</div>
<div>
<img src="readme/box_dewpoint.png" width="400"
 height= "300">
<img src="readme/box_humidity.png" width="400" height="300">
</div>
<div>
<img src="readme/box_cloud.png" width="400"
 height= "300">
<img src="readme/box_sunshine.png" width="400" height="300">
</div>
<div>
<img src="readme/box_winddirection.png" width="400"
 height= "300">
<img src="readme/box_windspeed.png" width="400" height="300">
</div>

<hr>

## :robot: Machine Learning Model

### Preprocessing

- **StandardScaler**: Used to standardize the features by removing the mean and scaling to unit variance.
- **PCA**: Applied to reduce the dimensionality of the dataset to 10 principal components.

### Model

- **Algorithm**: Random Forest Classifier
- **Reason for Selection**: Demonstrated superior performance and faster training time compared to other models such as Logistic Regression and Support Vector Classifier (SVC).
- **Hyperparameter Tuning**: Utilized GridSearchCV to find the optimal hyperparameters with 5-fold cross-validation.
  - **Hyperparameters**:
    - `n_estimators`: [300]
    - `criterion`: ["gini", "entropy"]
    - `max_depth`: [None, 10, 20, 30]
    - `min_samples_split`: [2, 5, 10]
    - `min_samples_leaf`: [1, 2, 4]

### Model Evaluation

- **Metrics**:
  - **Accuracy**: Evaluated on the test set.
  - **Confusion Matrix**: Generated to visualize the performance of the classification.
  - **ROC Curve**: Plotted to show the trade-off between true positive rate and false positive rate.
  - **AUC (Area Under Curve)**: Calculated to summarize the ROC curve.
  - **Threshold**: Evaluated different thresholds to optimize the recall and minimize false negatives.

### Results

- **Test Accuracy**: Achieved a test accuracy of 0.85.
- **Test Accuracy - Cross Validation**: Achieved a test accuracy of 0.84.
- **ROC Curve Area**: 0.88.
- **ROC Curve Area Cross Validation**: 0.86
- **Optimal Threshold**: 0.56

<div>
<img src="readme/confusion_matrix.png" width="500">
<img src="readme/confusion_matrix_optimal.png" width="500" height ="326">
</div>