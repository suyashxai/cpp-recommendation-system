# Crop Recommendation System — Project Report

**ML Internship Project**

---

## Table of Contents

1. [Introduction](#chapter-1--introduction)
2. [Problem Statement](#chapter-2--problem-statement)
3. [Objectives](#chapter-3--objectives)
4. [Existing System & Literature](#chapter-4--existing-system--literature)
5. [Proposed System](#chapter-5--proposed-system)
6. [System Requirements](#chapter-6--system-requirements)
7. [System Architecture](#chapter-7--system-architecture)
8. [Dataset](#chapter-8--dataset)
9. [Methodology](#chapter-9--methodology)
10. [Algorithms](#chapter-10--algorithms)
11. [Results](#chapter-11--results)
12. [Application Screens](#chapter-12--application-screens)
13. [Testing](#chapter-13--testing)
14. [Advantages](#chapter-14--advantages)
15. [Limitations](#chapter-15--limitations)
16. [Future Scope](#chapter-16--future-scope)
17. [Conclusion](#chapter-17--conclusion)
18. [References](#chapter-18--references)

---

## Chapter 1 — Introduction

### Agriculture and Crop Selection

Agriculture is the backbone of economies worldwide, particularly in developing nations. One of the most critical decisions a farmer makes each season is selecting which crop to cultivate. This decision affects not only the farmer's income but also food security at a regional and national level.

Effective crop selection requires understanding a complex combination of factors: soil nutrient levels, soil pH, local temperature, relative humidity, and rainfall patterns. Historically, farmers have relied on generalised recommendations, local knowledge passed down through generations, and basic soil testing. While valuable, these approaches often fail to account for the multi-factor interactions that determine crop suitability.

### Importance of Soil and Environmental Parameters

Research in agronomy has established that crop performance is influenced by multiple interdependent factors:

- **Nitrogen (N)**: Promotes leaf and stem growth; deficiency limits yield.
- **Phosphorus (P)**: Critical for root development and energy transfer.
- **Potassium (K)**: Improves drought resistance, disease tolerance, and fruit quality.
- **Temperature**: Each crop species has an optimal temperature range for germination, growth, and reproduction.
- **Humidity**: Affects transpiration, disease pressure, and water availability.
- **Soil pH**: Influences nutrient availability; most crops prefer slightly acidic to neutral soil.
- **Rainfall**: Determines water availability; different crops require dramatically different amounts.

The challenge is that no single parameter determines crop suitability — it is the combination of all seven parameters that matters. This is precisely the kind of multi-variable pattern recognition task at which machine learning excels.

### Role of Machine Learning

Machine learning algorithms can identify complex, non-linear patterns in historical agricultural data. When trained on a labelled dataset containing soil and environmental measurements alongside successful crop outcomes, an ML model learns the characteristic "fingerprint" of conditions under which each crop thrives. Given new measurements, the model predicts the most suitable crop based on those learned patterns.

### Project Motivation

The goal of this project is to demonstrate the practical application of machine learning to agricultural decision-making by building a complete, deployable web application. The system is designed to be accessible to farmers and agricultural extension workers without requiring ML expertise — a user simply enters field measurements and receives an instant recommendation.

---

## Chapter 2 — Problem Statement

In precision agriculture, matching the right crop to specific soil and environmental conditions is essential for maximising yield and sustainability. However, farmers — especially smallholders — typically lack access to sophisticated tools that can interpret multiple soil and environmental measurements simultaneously.

**The problem:** Given seven measurable agricultural parameters — Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Soil pH, and Rainfall — determine the most suitable crop to cultivate.

This is a supervised multi-class classification problem. The input space is seven-dimensional (real-valued), and the output is one of 22 discrete crop classes.

---

## Chapter 3 — Objectives

1. Build a machine learning classification system capable of recommending the most suitable crop from 22 classes based on seven agricultural input parameters.
2. Perform exploratory data analysis to understand dataset characteristics, class distribution, and feature correlations.
3. Train and evaluate multiple ML classification algorithms: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbours.
4. Compare model performance using accuracy, precision, recall, and F1-score metrics.
5. Select and save the best-performing model for deployment.
6. Deploy the trained model through a Flask web application with a responsive frontend.
7. Provide confidence scores with each recommendation.
8. Store and display prediction history using SQLite.
9. Implement input validation and error handling throughout the system.
10. Produce automated tests covering the model, API, and validation logic.

---

## Chapter 4 — Existing System & Literature

### Traditional Approaches

Traditional crop selection methods include:

- **Experience-based decisions**: Farmers select crops based on what has worked in previous seasons. This approach is vulnerable to changing climate conditions and does not adapt to soil degradation over time.
- **Extension service recommendations**: Agricultural extension officers provide general guidance based on regional averages. These recommendations may not account for individual field variations.
- **Basic soil testing**: Chemical soil tests measure nutrient levels, but the interpretation often relies on lookup tables that do not account for the combined effects of temperature, humidity, and rainfall.
- **Manual decision rules**: Some advisory systems use rule-based logic (e.g., "if pH < 5.5, recommend acidic-tolerant crops"), which cannot capture the complex multi-factor interactions in real fields.

### Limitations of Existing Approaches

- Individual field variability is not captured by regional averages.
- Multi-parameter interactions are rarely considered simultaneously.
- Rule-based systems require domain expert knowledge to build and maintain.
- Recommendations are rarely accompanied by confidence levels.
- These approaches do not improve over time with new data.

### Machine Learning Approaches in Agriculture

The application of ML to crop recommendation has been explored in academic literature. Researchers have applied Decision Trees, Random Forests, Support Vector Machines, and neural networks to crop selection problems with promising results. The Crop Recommendation Dataset used in this project (Ingle, 2020, Kaggle) has been used in several published analyses, demonstrating that ensemble methods such as Random Forest consistently achieve high accuracy on this classification task.

---

## Chapter 5 — Proposed System

### System Overview

The proposed system is an end-to-end web application that accepts seven agricultural input parameters from a user and returns a crop recommendation with an associated confidence score.

The system consists of three layers:

1. **Frontend Layer**: A responsive HTML/CSS/JavaScript interface accessible through a web browser. The user enters field measurements, submits the form, and receives an immediate recommendation.

2. **Backend Layer**: A Flask (Python) web server that receives requests, validates inputs, invokes the ML model, saves the result to SQLite, and returns the response.

3. **ML Layer**: A pre-trained Random Forest classifier (or the best-performing model identified during evaluation) loaded from a serialised `.pkl` file. The same `StandardScaler` used during training is applied to inputs at inference time to ensure consistent preprocessing.

### Key Design Decisions

- **Pre-trained model**: The model is trained once and saved. The Flask application loads the saved model at startup, avoiding retraining on every request.
- **Consistent preprocessing**: The `StandardScaler` fitted on the training data is saved alongside the model and applied identically during inference.
- **Feature order preservation**: The feature vector is always constructed in the same order as used during training to prevent silent prediction errors.
- **Parameterised SQL**: All database operations use parameterised queries to prevent SQL injection.

---

## Chapter 6 — System Requirements

### Hardware Requirements

| Component | Minimum |
|---|---|
| Processor | Intel Core i3 (or equivalent) |
| RAM | 4 GB |
| Storage | 500 MB free (for dataset, model, dependencies) |
| Network | Required for initial package installation |

### Software Requirements

| Software | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Primary language |
| Flask | 3.x | Web framework |
| Scikit-learn | 1.4.x | ML algorithms |
| Pandas | 2.x | Data loading and processing |
| NumPy | 1.26.x | Numerical computation |
| Matplotlib / Seaborn | Latest | Visualisation |
| Joblib | 1.4.x | Model serialisation |
| SQLite | Built-in | Prediction history storage |
| pytest | 8.x | Automated testing |
| HTML5 / CSS3 / JS | — | Frontend |

---

## Chapter 7 — System Architecture

### Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                      USER                           │
│              (Web Browser)                          │
└────────────────────┬────────────────────────────────┘
                     │  HTTP Request
                     ▼
┌─────────────────────────────────────────────────────┐
│                   FRONTEND                          │
│    HTML5 / CSS3 / JavaScript                        │
│    - Prediction Form                                │
│    - Dashboard                                      │
│    - History Table                                  │
│    - About Page                                     │
└────────────────────┬────────────────────────────────┘
                     │  POST /predict  or  POST /api/predict
                     ▼
┌─────────────────────────────────────────────────────┐
│                FLASK BACKEND                        │
│    app.py  →  Blueprint Routes                      │
│    ├── main_routes.py (/, /about, /history)         │
│    └── predict_routes.py (/predict, /api/predict)   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              INPUT VALIDATION                       │
│    backend/utils/validation.py                      │
│    - Required fields check                          │
│    - Numeric type coercion                          │
│    - Range validation                               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│               PREPROCESSING                         │
│    backend/services/model_service.py                │
│    - Load StandardScaler (scaler.pkl)               │
│    - Scale features                                 │
│    - Ensure correct feature order                   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                  ML MODEL                           │
│    Random Forest Classifier (crop_model.pkl)        │
│    - predict()                                      │
│    - predict_proba()                                │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              CROP PREDICTION                        │
│    recommended_crop + confidence score              │
│    + per-class probabilities                        │
└────────────┬───────────────────────┬────────────────┘
             │                       │
             ▼                       ▼
  ┌──────────────────┐   ┌───────────────────────┐
  │  SQLite Storage  │   │   HTTP Response        │
  │  (history)       │   │   (HTML or JSON)       │
  └──────────────────┘   └───────────────────────┘
```

### Data Flow

1. User enters values in the browser form.
2. JavaScript validates ranges client-side; form is submitted via HTTP POST.
3. Flask receives the form data and passes it to the validation module.
4. Validated data is passed to the model service, which loads the saved model and scaler.
5. The feature vector is scaled and passed to `model.predict()` and `model.predict_proba()`.
6. The prediction and confidence are saved to SQLite.
7. The result is rendered in the browser.

---

## Chapter 8 — Dataset

### Source

**Crop Recommendation Dataset** by Atharva Ingle, available on Kaggle:  
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

### Description

The dataset contains measurements collected under various agricultural conditions and annotated with the crop that is recommended for those conditions.

| Property | Value |
|---|---|
| Total Records | 2,200 |
| Features | 7 (all numerical) |
| Target Variable | `label` (crop name) |
| Crop Classes | 22 |
| Missing Values | None |
| File Format | CSV |

### Features

| Feature | Description | Unit |
|---|---|---|
| N | Nitrogen content | kg/ha |
| P | Phosphorus content | kg/ha |
| K | Potassium content | kg/ha |
| temperature | Average temperature | °C |
| humidity | Relative humidity | % |
| ph | Soil pH | — |
| rainfall | Annual rainfall | mm |

### Target Variable

The `label` column contains one of 22 crop names:  
apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, motherbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon.

### Class Distribution

Each of the 22 classes contains exactly 100 samples, making the dataset balanced. This is important because balanced classes allow straightforward accuracy evaluation and do not require oversampling or class-weight correction.

### Data Preprocessing

The following preprocessing steps are applied:

1. **Duplicate removal**: Duplicate rows (if any) are removed to prevent data leakage.
2. **Feature/target separation**: The seven feature columns are separated from the `label` column.
3. **Train/test split**: 80% of data is used for training, 20% for evaluation. Stratified splitting preserves class proportions.
4. **Standard scaling**: All features are standardised to zero mean and unit variance. This is required for Logistic Regression and KNN, and is applied consistently to all models to ensure identical preprocessing at inference time.

---

## Chapter 9 — Methodology

### Pipeline

```
Data Collection
    │  Load Crop_recommendation.csv using Pandas
    ▼
Data Cleaning
    │  Check for missing values (none found)
    │  Remove duplicate rows
    ▼
EDA
    │  Class distribution, statistical summary
    │  Correlation heatmap, feature distributions, boxplots
    │  Figures saved to reports/figures/
    ▼
Preprocessing
    │  Feature/target separation
    │  Stratified train/test split (80/20)
    │  StandardScaler fitted on training data
    ▼
Model Training
    │  Logistic Regression (max_iter=1000)
    │  Decision Tree (CART)
    │  Random Forest (100 estimators)
    │  K-Nearest Neighbours (k=5)
    ▼
Model Evaluation
    │  Accuracy, Precision, Recall, F1-score (weighted)
    │  Confusion matrix for best model
    │  Feature importance plot (Random Forest)
    ▼
Model Selection
    │  Best model selected by highest weighted F1-score
    ▼
Model Saving
    │  joblib.dump(model, 'model/crop_model.pkl')
    │  joblib.dump(scaler, 'model/scaler.pkl')
    ▼
Flask Deployment
    │  Model loaded at startup
    │  Scaler applied to every prediction request
    │  Results served via HTML and JSON endpoints
```

### Why Weighted F1-score for Model Selection?

Although the dataset is balanced, weighted F1-score is chosen as the primary selection metric because:
- It combines precision and recall into a single score.
- Weighting by class support gives a fair representation of overall performance.
- It is robust to any residual class imbalances that may appear in the test split.

---

## Chapter 10 — Algorithms

### 10.1 Logistic Regression

**Concept**: Logistic Regression is a linear classifier that models the probability of each class using the logistic (sigmoid) function. For multi-class problems, the One-vs-Rest or Softmax strategy is applied.

**Working**: The model learns weights for each feature. During prediction, the dot product of weights and features is passed through the softmax function to produce class probabilities.

**Advantages**:
- Simple and interpretable
- Computationally efficient
- Works well when decision boundaries are linear
- Probability outputs are well-calibrated

**Limitations**:
- Assumes linear separability
- Cannot capture complex non-linear relationships in the data
- Performance degrades when features interact non-linearly

---

### 10.2 Decision Tree

**Concept**: A Decision Tree recursively partitions the feature space by selecting the feature and threshold that best separates the classes at each node, using metrics such as Gini impurity or Information Gain.

**Working**: Starting from the root node, the model splits data at each node based on the feature that maximally reduces impurity. Prediction traverses from root to leaf and returns the majority class at the leaf.

**Advantages**:
- Highly interpretable — the tree can be visualised and explained
- Handles non-linear relationships naturally
- No feature scaling required
- Fast prediction

**Limitations**:
- Prone to overfitting on training data without pruning
- Sensitive to small changes in data (high variance)
- A single tree may not generalise well

---

### 10.3 Random Forest

**Concept**: Random Forest is an ensemble method that trains many Decision Trees on random subsets of the data (bootstrap samples) and random subsets of features at each split. The final prediction is the majority vote across all trees.

**Working**: 
1. For each of the `n_estimators` trees: draw a bootstrap sample from the training data.
2. At each node split, consider a random subset of features (√n for classification).
3. Build the tree to full depth.
4. For prediction, each tree votes; the majority class wins.

**Advantages**:
- Reduces variance through averaging (bagging)
- Highly accurate on structured/tabular data
- Provides feature importance scores
- Resistant to overfitting compared to individual trees
- Handles missing values and maintains accuracy with many features

**Limitations**:
- Less interpretable than a single tree
- Computationally more expensive than simple models
- Memory usage scales with number of trees

**Why Random Forest is the preferred model in this project:**  
On the Crop Recommendation Dataset, Random Forest consistently achieves the highest weighted F1-score among the four candidates, typically exceeding 99% accuracy. Its feature importance output also provides insight into which agricultural parameters most influence the recommendation.

---

### 10.4 K-Nearest Neighbours (KNN)

**Concept**: KNN is a non-parametric, instance-based learning algorithm. Classification is performed by finding the `k` training samples closest to the query point and assigning the majority class among those neighbours.

**Working**: Distance (typically Euclidean) is computed between the query point and all training samples. The `k` nearest neighbours vote, and the majority class is returned.

**Advantages**:
- Simple concept, easy to understand
- No training phase (lazy learner)
- Naturally handles multi-class problems
- Can adapt to complex decision boundaries with enough data

**Limitations**:
- Prediction is slow for large datasets (must compute distances to all training points)
- Sensitive to irrelevant features and scale (feature scaling is required)
- Performance depends heavily on the choice of k
- High memory usage for large training sets

---

## Chapter 11 — Results

> **Note**: The exact values below are generated during training and saved to `reports/figures/evaluation_summary.txt`. The typical values on this dataset are shown for reference. Run `python model/train_model.py` to generate your actual results.

### Model Comparison (Typical Results)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | ~96.1% | ~96.2% | ~96.1% | ~96.0% |
| Decision Tree | ~98.2% | ~98.2% | ~98.2% | ~98.2% |
| **Random Forest** | **~99.3%** | **~99.3%** | **~99.3%** | **~99.3%** |
| KNN | ~97.5% | ~97.5% | ~97.5% | ~97.5% |

**Selected Model**: Random Forest (highest weighted F1-score)

### Evaluation Notes

- All models perform well on this dataset, reflecting its clean structure.
- Random Forest outperforms the other three models due to its ensemble nature.
- The Confusion Matrix for the best model is saved at `reports/figures/06_confusion_matrix.png`.
- Feature importances are saved at `reports/figures/07_feature_importance.png`.
- The model comparison bar chart is saved at `reports/figures/05_model_comparison.png`.

### Generated Figures

| File | Description |
|---|---|
| `01_crop_distribution.png` | Bar chart of crop counts |
| `02_correlation_heatmap.png` | Feature correlation heatmap |
| `03_feature_distributions.png` | Histogram for each feature |
| `04_feature_boxplots.png` | Boxplot for each feature |
| `05_model_comparison.png` | Accuracy/Precision/Recall/F1 bar chart |
| `06_confusion_matrix.png` | Confusion matrix of best model |
| `07_feature_importance.png` | Random Forest feature importances |
| `evaluation_summary.txt` | Text summary of all model metrics |

---

## Chapter 12 — Application Screens

### Home / Dashboard
The home page displays a live dashboard with:
- Total predictions made
- Most frequently recommended crop
- Average confidence score
- Number of supported crop classes

Below the dashboard, the prediction form is displayed.

### Prediction Form
Seven labelled input fields with placeholder values, range hints, and client-side validation. A "Get Recommendation" button submits the form.

### Prediction Result
Displays the recommended crop name prominently, along with:
- Confidence percentage with an animated progress bar
- Top 5 crop probabilities with bar visualisation
- Input summary table

### History Page
Tabular display of all previous predictions with colour-coded confidence badges. Supports up to 100 most recent predictions.

### About Page
Explains the project purpose, features, technology stack, ML workflow, dataset source, and disclaimer.

---

## Chapter 13 — Testing

### Unit Testing — Validation Module

| Test | Expected Result |
|---|---|
| Valid input passes | `is_valid = True` |
| Missing field `N` | `is_valid = False`, error mentions `N` |
| Non-numeric value | `is_valid = False` |
| pH below 0 | `is_valid = False` |
| Temperature above 50 | `is_valid = False` |
| Empty dict | `is_valid = False` |
| None input | `is_valid = False` |

### API Testing

| Endpoint | Test | Expected |
|---|---|---|
| `GET /api/health` | Normal request | 200, `status: ok` |
| `GET /` | Normal request | 200 |
| `GET /about` | Normal request | 200 |
| `GET /history` | Normal request | 200 |
| `POST /api/predict` | Valid payload | 200, `success: true` |
| `POST /api/predict` | Missing `N` | 400, `success: false` |
| `POST /api/predict` | Non-numeric temperature | 400 |
| `POST /api/predict` | Negative rainfall | 400 |
| `POST /api/predict` | pH = 15 | 400 |
| `POST /api/predict` | Empty body | 400 |

### Model Testing

| Test | Expected Result |
|---|---|
| Model file exists | `True` |
| Scaler file exists | `True` |
| Model loads without error | No exception |
| Model expects 7 features | `n_features_in_ == 7` |
| Prediction for valid input | Returns a crop name |
| Predicted crop is a known crop | In the 22-class set |
| `predict_proba` available | `True` |
| Probabilities sum to 1 | Within 1e-4 |

### Running Tests

```bash
pytest tests/ -v
```

---

## Chapter 14 — Advantages

1. **Multi-factor analysis**: The system simultaneously considers all seven agricultural parameters, capturing interactions that simple rule-based systems miss.

2. **Data-driven**: Recommendations are based on patterns learned from 2,200 agricultural records, not arbitrary heuristics.

3. **Confidence score**: Users receive not just a crop name but also a probability estimate, helping them understand how confident the model is.

4. **Accessible interface**: No ML knowledge is required to use the system — any user who can measure soil nutrients and look up local weather data can use it.

5. **REST API**: The prediction endpoint can be called programmatically, enabling integration with other agricultural software.

6. **History tracking**: All predictions are stored, allowing users to track recommendations over time and across field locations.

7. **Transparent evaluation**: Four algorithms are compared and the selection rationale is documented, rather than simply choosing the most popular model.

---

## Chapter 15 — Limitations

1. **Dataset constraints**: The dataset contains 2,200 records covering 22 crops. Crops not in this list cannot be recommended.

2. **Static training data**: The model does not retrain automatically as new data becomes available. Changes in climate patterns or new crop varieties require retraining.

3. **No geographical awareness**: The system does not account for regional soil types, altitude, or local microclimates.

4. **Yield is not predicted**: The system recommends a crop suitable for given conditions but does not predict the expected yield.

5. **Point-in-time measurements**: The model is based on average conditions; it does not account for seasonal variations within a growing period.

6. **No integration with real sensors**: Currently, users must manually enter measurements. Integration with soil sensors or weather stations would improve usability.

7. **Model interpretability**: While feature importances are provided, the model does not explain *why* a specific crop is recommended in human-understandable terms for each individual prediction.

---

## Chapter 16 — Future Scope

1. **Weather API integration**: Automatically retrieve real-time temperature, humidity, and rainfall from a weather service (e.g., OpenWeatherMap) based on the user's location.

2. **Soil sensor integration**: Connect to IoT soil sensors to auto-fill N, P, K, and pH values.

3. **Location-based recommendations**: Use GPS coordinates to select region-appropriate models trained on local datasets.

4. **Explainable AI (XAI)**: Integrate SHAP or LIME to generate per-prediction feature-level explanations.

5. **Larger datasets**: Retrain with larger, geographically diverse datasets to improve generalisation.

6. **Advanced models**: Evaluate Gradient Boosting (XGBoost, LightGBM) and neural networks, which may improve performance on more complex datasets.

7. **Mobile application**: A native iOS/Android application with offline capability would make the system accessible to farmers without reliable internet.

8. **Multilingual interface**: Support regional languages to increase accessibility in rural communities.

9. **Crop yield prediction**: Extend the system to predict expected yield in addition to crop suitability.

10. **Market price integration**: Incorporate crop market prices to recommend the most economically beneficial crop among suitable options.

---

## Chapter 17 — Conclusion

This project successfully demonstrates the end-to-end application of machine learning to the agricultural crop recommendation problem. Four classification algorithms were trained and evaluated on the Crop Recommendation Dataset: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbours.

Random Forest achieved the highest weighted F1-score, consistent with its reputation as a strong baseline for structured tabular data. The selected model was serialised and deployed through a Flask web application, providing a responsive and intuitive interface that is accessible without any machine learning expertise.

The system implements the complete ML pipeline — from data loading and EDA through preprocessing, training, evaluation, model saving, and deployment — in a modular, well-documented codebase appropriate for an internship project. Input validation, error handling, automated tests, prediction history storage, and REST API access demonstrate production-oriented software engineering practices.

The project establishes a solid foundation for future improvements, including real-time data integration, explainability features, and extended crop coverage. It serves as a practical example of how machine learning can provide tangible value in agricultural decision support.

---

## Chapter 18 — References

1. **Crop Recommendation Dataset**  
   Ingle, A. (2020). *Crop Recommendation Dataset*. Kaggle.  
   https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

2. **Scikit-learn: Machine Learning in Python**  
   Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.  
   https://scikit-learn.org/stable/

3. **Flask Documentation**  
   Pallets Projects. *Flask 3.x Documentation*.  
   https://flask.palletsprojects.com/

4. **Pandas Documentation**  
   The Pandas Development Team. *pandas: Powerful data structures for data analysis*.  
   https://pandas.pydata.org/docs/

5. **NumPy Documentation**  
   Harris, C.R., et al. (2020). *Array programming with NumPy*. Nature, 585, 357–362.  
   https://numpy.org/doc/

6. **Matplotlib Documentation**  
   Hunter, J.D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering, 9(3), 90–95.  
   https://matplotlib.org/stable/

7. **Seaborn Documentation**  
   Waskom, M.L. (2021). *seaborn: statistical data visualization*. Journal of Open Source Software, 6(60), 3021.  
   https://seaborn.pydata.org/

8. **Random Forests**  
   Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

9. **Decision Trees (CART)**  
   Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.

10. **pytest Documentation**  
    https://docs.pytest.org/en/stable/

---

*This report was produced as part of an ML internship project.*
