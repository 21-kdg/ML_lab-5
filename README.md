# ML Practical 5: Diabetes Prediction Using K-Nearest Neighbors (KNN)

## Student Details
- **Name:** Krishna Gupta
- **Roll Number:** 10
- **Section-Batch:** B - A1
- **Experiment:** Lab 5

## 1. Aim
To develop a machine learning model using the K-Nearest Neighbors (KNN) algorithm to predict diabetes outcomes based on medical features. The experiment also includes data preprocessing, feature scaling, cross-validation, and hyperparameter tuning to evaluate and optimize model performance.

## 2. Dataset Description
The experiment uses the Diabetes dataset containing 768 records and 9 columns.

The dataset includes eight input features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

**Target variable:** `Outcome`
- `0` – No Diabetes
- `1` – Diabetes

## 3. Technologies and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## 4. Methodology

1. Load the diabetes dataset using Pandas.
2. Inspect the dataset dimensions, columns, data types, and statistical summaries.
3. Check for missing values and duplicate records.
4. Identify invalid zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI.
5. Replace invalid zeros with NaN and fill the missing values using the median of each feature.
6. Separate the input features (`X`) and target variable (`y`).
7. Split the dataset into 80% training data and 20% testing data using stratified sampling.
8. Apply StandardScaler to standardize the training and testing features.
9. Train the initial KNN classifier with `n_neighbors=5`.
10. Evaluate the model using accuracy, precision, recall, F1-score, classification report, and confusion matrix.
11. Compare training and testing accuracy for K values ranging from 1 to 30.
12. Use five-fold StratifiedKFold cross-validation to select a suitable K value.
13. Apply GridSearchCV to optimize KNN hyperparameters.
14. Evaluate the optimized classifier on the test dataset and compare the results.

## 5. Model Used

**K-Nearest Neighbors (KNN)** is a supervised machine learning algorithm that classifies a sample based on the classes of its nearest training observations.

The initial model uses:
- Number of neighbors: 5
- Distance-based classification
- Standardized input features

The optimized model uses the best hyperparameters identified by GridSearchCV.

## 6. Results

### Initial KNN Model (K = 5)

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 75.32% |
| Precision | 66.00% |
| Recall | 61.11% |
| F1-score | 63.46% |

### K-Value Selection
- **Best K based on test accuracy:** 8
- **Best test accuracy:** 76.62%
- **Best cross-validation accuracy during K-value search:** 77.04%

### GridSearchCV Results
Best hyperparameters:
- `n_neighbors = 19`
- `metric = euclidean`
- `p = 1`
- `weights = uniform`

Best cross-validation accuracy: **77.69%**

### Final Optimized Model

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 74.68% |
| Precision | 66.67% |
| Recall | 55.56% |
| F1-score | 60.61% |

The optimized model achieved the highest cross-validation score among the tested GridSearchCV configurations, but its test accuracy and recall were lower than those of the initial model. This demonstrates that the best cross-validation configuration does not necessarily produce the best results on a separate test set.

## 7. Visualizations
The notebook includes the following visualizations:
- Box plot for examining feature distributions and outliers.
- Histogram for understanding numerical feature distributions.
- Count plot for observing class frequencies.
- Scatter plot for exploring relationships between features.
- Heatmap for visualizing the confusion matrix.
- Pair plot for examining relationships between multiple features.
- Accuracy versus K-values graph for comparing training and testing performance.

## 8. Conclusion
The diabetes dataset was preprocessed by handling invalid zero measurements and applying median imputation. The features were standardized before training the KNN classifier. Different values of K were evaluated, and GridSearchCV was used to optimize the hyperparameters. The experiment demonstrates the importance of preprocessing, cross-validation, and multiple evaluation metrics when developing a classification model.

The resulting classifier is intended for educational experimentation and should not be used as a substitute for professional medical diagnosis.

## 9. How to Run
1. Open the notebook `ML_Practical_5_.ipynb` in Google Colab or Jupyter Notebook.
2. Upload the diabetes CSV dataset required by the notebook.
3. Install the required libraries if they are not already available.
4. Run the notebook cells in sequence.
5. Review the evaluation metrics, confusion matrix, accuracy graph, and GridSearchCV results.

## 10. Files
- `ML_Practical_5_.ipynb` – Notebook containing the preprocessing, model training, evaluation, and hyperparameter tuning.
- `diabetes (6).csv` – Dataset required to run the experiment.

