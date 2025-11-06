# Olist Customer Satisfaction Prediction

**An End-to-End Machine Learning Project**

This project analyzes the Olist E-commerce dataset to predict customer satisfaction. The primary goal is to build a binary classification model that can identify whether a customer will leave a positive review (4-5 stars) or a negative review (1-3 stars) based on their complete order data.

This model can help Olist proactively identify at-risk customers, allowing the business to intervene and improve customer experience, with a focus on logistics and service.

## Dataset

This project uses a relational e-commerce dataset spread across 9 separate `.csv` files:

- `olist_customers_dataset.csv`
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_products_dataset.csv`
- `olist_sellers_dataset.csv`
- `product_category_name_translation.csv`
- `olist_geolocation_dataset.csv`

## Project Pipeline

The project is broken down into 7 distinct phases:

### Phase 1: Data Loading & Definition

- Loaded all 9 datasets into pandas DataFrames.
- Inspected initial data types, column names, and missing values.

### Phase 2: Data Merging & Aggregation

- Handled the complex relational data by aggregating tables with one-to-many relationships (`olist_order_items` and `olist_order_payments`).
- Merged all 6 key tables into a single master DataFrame, with each row representing a single, unique order.

### Phase 3: Exploratory Data Analysis (EDA)

A deep dive into the merged data to understand trends and find key relationships.

- **Target Variable Analysis**: Analyzed `review_score` and found a severe class imbalance, with over 75% of reviews being positive (4 or 5 stars). This confirmed that accuracy would be a poor metric and that F1-Score and Recall for the negative class are the primary evaluation metrics.
- **Time Series Analysis**: Plotted order volume and revenue over time, identifying seasonal peaks (e.g., Black Friday).
- **Geographic Analysis**: Plotted top customer and seller states, revealing that the business is heavily concentrated in São Paulo (SP). A folium map was generated to visualize the logistical challenge of sellers in the southeast shipping to customers across Brazil.
- **Bivariate Analysis**: Used box plots to confirm a strong negative correlation between delivery time and review score.
- **Text & Correlation**: Generated a WordCloud from review comments and a seaborn heatmap, both of which confirmed that delivery-related features are the most important.

### Phase 4: Feature Engineering

Created new, high-impact features from the existing data to improve model performance.

- `target_satisfied`: The binary target variable (1-3 stars = 0, 4-5 stars = 1).
- `delivery_time_days`: Total time in days from purchase to delivery.
- `estimated_vs_actual_days`: A key feature showing the difference between the estimated and actual delivery date. A negative number means the order was late.
- `processing_time_days`: Time taken by the seller to hand over the order to the carrier.
- `is_same_state`: A binary flag (1 or 0) indicating if the customer and seller are in the same state.
- `product_volume_cm3`: `length * width * height` to create a single feature for product size.

### Phase 5: Pre-processing

Prepared the final, clean data for modeling using Scikit-Learn pipelines.

- Dropped rows with critical missing data (e.g., non-delivered orders).
- Built a `ColumnTransformer` pipeline to:
  - Impute missing numerical values with the median.
  - Scale all numerical features using `StandardScaler`.
  - Impute missing categorical values with the mode.
  - Encode all categorical features using `OneHotEncoder`.
- Split the data into training (80%) and testing (20%) sets, using `stratify=y` to ensure the class imbalance was preserved.

### Phase 6: Model Implementation

Trained and evaluated 10 different classification models to find the most promising algorithm.

- `DummyClassifier` (as a baseline)
- `LogisticRegression`
- `KNeighborsClassifier`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `AdaBoostClassifier`
- `GradientBoostingClassifier`
- `XGBClassifier`
- `GaussianNB`
- `SVC` (Linear)

The ensemble models (Random Forest, Gradient Boosting, and XGBoost) were the clear top performers.

### Phase 7: Hyperparameter Tuning & Final Evaluation

- Selected the best model from Phase 6 (XGBoost).
- Used `RandomizedSearchCV` to find the optimal hyperparameters, specifically optimizing for the F1-Score of the 'Not Satisfied' (Class 0) class.
- Evaluated the final, tuned model on the unseen test set to get the final project results.

## Final Model & Results

The final model, a tuned `XGBClassifier`, achieved the following performance on the unseen test set:

```
--- Classification Report (Test Set) ---
                   precision    recall  f1-score   support

Not Satisfied (0)       0.71      0.30      0.42      4002
    Satisfied (1)       0.84      0.97      0.90     15000

         accuracy                           0.83     19002
        macro avg       0.77      0.63      0.66     19002
     weighted avg       0.81      0.83      0.80     19002
```

### Key Metrics:

- **Best F1-Score (Class 0)**: 0.42
- **Precision (Class 0)**: 0.71
  - *Interpretation*: When the model predicts a customer is 'Not Satisfied', it is correct 71% of the time. This makes it a reliable model for business action.
- **Recall (Class 0)**: 0.30
  - *Interpretation*: The model successfully identified 30% of all actual 'Not Satisfied' customers. The model is "cautious," prioritizing accuracy over finding every single unhappy customer.

## Key Findings & Business Recommendation

The feature importance plot from the final model revealed the most significant drivers of customer satisfaction:

- **Delivery Punctuality** (`estimated_vs_actual_days`): This was the #1 most important feature. Customers are more frustrated by a late delivery (one that misses its estimate) than by a long delivery that arrives on time.
- **Total Delivery Time** (`delivery_time_days`): The overall speed of delivery was the second most important factor.
- **Processing Time & Freight**: Other factors like `processing_time_days` (how long the seller takes to ship) and `freight_value` (shipping cost) also had a notable impact.

### Recommendation:

To improve customer satisfaction, Olist should focus its efforts on logistics. The two most impactful, data-driven actions would be:

1. Set Accurate Delivery Estimates.
2. Reduce Actual Delivery Times.

## How to Run This Project

1. Clone this repository:
   ```bash
   git clone [YOUR_GITHUB_REPO_URL]
   ```

2. Place all 9 `.csv` files from the Olist dataset in the `dataset/` directory (already structured in this repo).

3. Install the required libraries (including Jupyter to run the notebook):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost wordcloud folium jupyter nbconvert
   ```

4. Run the notebook `ML_Project_Final.ipynb`:
   - Open interactively:
     ```bash
     jupyter notebook ML_Project_Final.ipynb
     ```
   - Or execute all cells non-interactively and produce an executed copy:
     ```bash
     jupyter nbconvert --to notebook --execute ML_Project_Final.ipynb --output ML_Project_Final_executed.ipynb
     ```

**Note**: End-to-end execution may take 20–30 minutes due to model training (especially SVM in Phase 6) and hyperparameter tuning (Phase 7).