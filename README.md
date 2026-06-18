# GCI World 2026 Spring Final Assignment

## Telecom Churn Prediction and Retention Strategy

## 1. Project Overview

This project analyzes historical telecom customer data from Company A, an anonymous telecommunications service provider. The objective is to identify customers who are likely to churn within the 31–60 day prediction window and propose a data-driven retention strategy based on exploratory data analysis and machine learning.

The final business proposal focuses on building an AI-powered churn risk scoring system that helps Company A prioritize high-risk customers and apply targeted retention actions instead of using broad, costly discount campaigns.

---

## 2. Business Problem

Company A faces a major customer retention issue. The dataset shows that nearly half of the observed customers churned.

* Total customers analyzed: 100,000
* Churned customers: 49,562
* Churn rate: 49.56%

This suggests that churn is not a minor operational issue. It represents a significant revenue-risk problem that requires proactive intervention.

The key business question is:

> How can Company A identify customers who are most likely to churn and prioritize retention actions before revenue is lost?

---

## 3. Dataset Description

The dataset consists of two CSV files:

### 3.1 Client.csv

This file contains customer profile and account-level information.

* Rows: 100,000
* Columns: 50

Examples of variables include:

* Customer tenure and household subscription data
* Total calls and total minutes of use
* Total and adjusted revenue
* Handset information
* Area and account characteristics
* Demographic and household indicators

### 3.2 Record.csv

This file contains usage behavior and churn information.

* Rows: 100,000
* Columns: 51

Examples of variables include:

* Mean monthly revenue
* Mean monthly minutes of use
* Change in usage and revenue
* Dropped and blocked calls
* Customer care interactions
* Churn label
* Customer_ID

The target variable is:

```text
churn
```

Where:

```text
0 = Customer did not churn
1 = Customer churned
```

The churn event represents whether the customer churned between 31–60 days after the observation date.

---

## 4. Project Structure

```text
GCI_Final_Assignment/
│
├── data/
│   ├── Client.csv
│   └── Record.csv
│
├── notebook/
│   └── GCI_Churn_Project.ipynb
│
├── outputs/
│   ├── charts/
│   └── tables/
│
├── slides/
│   └── final_presentation.pdf
│
└── README.md
```

---

## 5. Data Validation

Before analysis, both datasets were checked for data consistency.

### Customer_ID Validation

Both files contain 100,000 unique Customer_ID values.

```text
Client.csv:
Rows: 100,000
Unique Customer_ID: 100,000
Duplicated Customer_ID: 0
Missing Customer_ID: 0

Record.csv:
Rows: 100,000
Unique Customer_ID: 100,000
Duplicated Customer_ID: 0
Missing Customer_ID: 0
```

All customer IDs matched between the two files.

```text
Client IDs: 100,000
Record IDs: 100,000
Common IDs: 100,000
Client IDs not in Record: 0
Record IDs not in Client: 0
```

Therefore, the two files were merged safely using a one-to-one merge on `Customer_ID`.

Final merged dataset:

```text
Merged dataframe shape: (100000, 100)
```

---

## 6. Missing Values Summary

Some customer demographic and household-related variables had high missing values.

Examples:

```text
numbcars: 49.37%
dwllsize: 38.31%
HHstatin: 37.92%
ownrent: 33.71%
dwlltype: 31.91%
lor: 30.19%
income: 25.44%
adults: 23.02%
infobase: 22.08%
```

These variables were treated carefully during modeling using preprocessing pipelines. Sensitive or potentially biased variables, such as `ethnic`, were excluded from the final business interpretation to reduce fairness and compliance risks.

---

## 7. Exploratory Data Analysis

The EDA focused on identifying business-relevant churn signals.

### 7.1 Churn Distribution

The churn rate was almost balanced:

```text
Not Churned: 50.44%
Churned: 49.56%
```

This indicates a major churn problem for Company A.

### 7.2 Monthly Revenue

Average monthly revenue was similar between churned and retained customers.

```text
Average monthly revenue - all customers: 58.72
Average monthly revenue - churned customers: 58.21
Average monthly revenue - not churned customers: 59.22
```

This means churn is not limited to low-value customers. Customers who churned still represented meaningful revenue.

### 7.3 Revenue at Risk

The annualized revenue associated with churned customers was estimated as:

```text
Monthly revenue at risk: 2,870,795.53
Annualized revenue at risk: 34,449,546.40
```

This shows that churn creates a significant revenue-risk issue for Company A.

### 7.4 Usage Decline

Customers who churned showed lower average usage and sharper decline in monthly usage.

Key finding:

```text
Not Churned average mou_Mean: 543.21
Churned average mou_Mean: 483.31
```

Also, `change_mou` showed a larger decline among churned customers.

```text
Not Churned change_mou: -5.34
Churned change_mou: -22.76
```

This suggests that declining usage is an early warning signal for churn.

### 7.5 Equipment Age

Customers who churned had older equipment on average.

```text
Not Churned eqpdays: 363.28
Churned eqpdays: 421.09
```

This suggests that device age may be connected to churn risk and can be used for targeted handset upgrade campaigns.

### 7.6 Network Quality

Dropped and blocked call metrics did not show higher averages among churned customers.

Example:

```text
drop_blk_Mean - Not Churned: 10.29
drop_blk_Mean - Churned: 9.80
```

This suggests that in this dataset, churn risk appears to be more strongly linked to usage decline and equipment age than dropped or blocked calls alone.

### 7.7 Customer Care

Churned customers had fewer customer-care interactions on average.

```text
custcare_Mean - Not Churned: 1.98
custcare_Mean - Churned: 1.60
```

This suggests that many at-risk customers may leave silently rather than actively complain.

---

## 8. Modeling Objective

The machine learning task was defined as a binary classification problem.

```text
Target variable: churn
Task type: Binary Classification
Business goal: Predict customers likely to churn within 31–60 days
```

The model was designed to support a retention strategy by ranking customers based on churn risk.

---

## 9. Feature Selection

A total of 48 features were selected for modeling.

```text
Number of selected features: 48
Numerical features: 40
Categorical features: 8
```

Feature groups included:

* Revenue features
* Usage features
* Network and call quality features
* Customer care features
* Tenure and device features
* Account and customer profile features

The model excluded `Customer_ID` because it is only an identifier.

---

## 10. Data Preprocessing

The preprocessing pipeline included:

### Numerical Features

* Missing values were imputed using the median.
* Features were scaled using StandardScaler.

### Categorical Features

* Missing values were imputed using the most frequent value.
* Categorical variables were encoded using OneHotEncoder.

### Train/Test Split

The dataset was split into training and testing sets.

```text
Training data: 80,000 rows
Testing data: 20,000 rows
Test size: 20%
Random state: 42
Stratified split: Yes
```

The churn distribution was preserved in both train and test sets.

```text
Train churn rate: 49.56%
Test churn rate: 49.56%
```

---

## 11. Machine Learning Models

Two models were built and compared:

1. Logistic Regression
2. Random Forest Classifier

### 11.1 Logistic Regression Baseline

The Logistic Regression model was used as a baseline.

```text
Model: Logistic Regression
Accuracy: 0.5919
Precision: 0.5877
Recall: 0.5913
F1-score: 0.5895
ROC-AUC: 0.6230
```

This model performed better than random prediction but had limited predictive power.

### 11.2 Random Forest Classifier

The Random Forest model was selected as the final model.

```text
Model: Random Forest Classifier
Accuracy: 0.6181
Precision: 0.6021
Recall: 0.6764
F1-score: 0.6371
ROC-AUC: 0.6696
```

Random Forest outperformed Logistic Regression across key evaluation metrics, especially recall.

Recall is important in this business problem because missing a high-risk churn customer can lead to preventable revenue loss.

---

## 12. Model Comparison

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.5919 |    0.5877 | 0.5913 |   0.5895 |  0.6230 |
| Random Forest       |   0.6181 |    0.6021 | 0.6764 |   0.6371 |  0.6696 |

The Random Forest model was selected as the final model because it achieved stronger overall performance and better recall for churn customers.

---

## 13. Feature Importance

The Random Forest model identified the following top churn prediction drivers:

| Rank | Feature     | Importance |
| ---: | ----------- | ---------: |
|    1 | eqpdays     |   0.117469 |
|    2 | months      |   0.092994 |
|    3 | change_mou  |   0.045341 |
|    4 | mou_Mean    |   0.043684 |
|    5 | hnd_price   |   0.035654 |
|    6 | totmrc_Mean |   0.033743 |
|    7 | adjrev      |   0.032738 |
|    8 | totrev      |   0.030334 |
|    9 | avg3mou     |   0.028095 |
|   10 | avgrev      |   0.025414 |

The main interpretation is that churn risk is strongly related to:

* Equipment age
* Customer tenure
* Usage decline
* Monthly usage level
* Revenue and customer value indicators

---

## 14. Risk Segmentation

The final model was used to calculate a churn risk score for each customer.

Customers were divided into three risk groups:

| Risk Segment | Customers | Actual Churn Rate | Avg Monthly Revenue | Annualized Revenue |
| ------------ | --------: | ----------------: | ------------------: | -----------------: |
| Low Risk     |    50,000 |            32.44% |               63.50 |      38,006,587.53 |
| Medium Risk  |    30,000 |            58.29% |               54.98 |      19,724,062.06 |
| High Risk    |    20,000 |            79.27% |               52.35 |      12,481,775.81 |

The model successfully concentrated churn risk into the top 20% highest-risk customer segment, where the actual churn rate reached 79.27%.

---

## 15. Business Proposal

The proposed solution is an:

```text
AI-Powered Churn Risk Scoring System
```

The system would score customers regularly and classify them into Low, Medium, and High Risk segments.

### Recommended Retention Strategy

| Risk Segment | Customer Signal                        | Recommended Action                                             |
| ------------ | -------------------------------------- | -------------------------------------------------------------- |
| High Risk    | Old equipment + high churn probability | Offer handset upgrade or device discount                       |
| High Risk    | Sharp decline in monthly usage         | Send personalized plan optimization or bonus usage offer       |
| Medium Risk  | Moderate churn probability             | Monitor usage and send light retention offer                   |
| Low Risk     | Low churn probability                  | Avoid costly retention offers and maintain standard engagement |

This allows Company A to move from mass discount campaigns to targeted retention actions.

---

## 16. Estimated Financial Impact

A risk-adjusted impact calculation was conducted using the High Risk segment.

High Risk segment:

```text
Customers: 20,000
Actual churn rate: 79.27%
Expected churners: approximately 15,854
```

Estimated impact:

| Retention Improvement | Expected Saved Customers | Risk-Adjusted Annual Revenue Protected |
| --------------------- | -----------------------: | -------------------------------------: |
| 5%                    |                      793 |                             497,976.93 |
| 10%                   |                    1,585 |                             995,953.87 |
| 15%                   |                    2,378 |                           1,493,930.80 |

A 10% retention improvement among predicted high-risk churners could protect approximately 996K in annual revenue.

---

## 17. Final Recommendation

Company A should implement a churn risk scoring system that prioritizes the highest-risk customers for targeted retention actions.

The recommended strategy is:

1. Score all customers using the Random Forest churn model.
2. Prioritize the top 20% high-risk customers.
3. Use customer-specific signals to assign retention actions:

   * Device upgrade offer for customers with old equipment.
   * Plan optimization or bonus usage offers for customers with declining usage.
   * Light-touch retention campaigns for medium-risk customers.
   * Avoid costly incentives for low-risk customers.
4. Monitor retention campaign results and retrain the model regularly.

This approach can help Company A reduce preventable churn, improve retention budget efficiency, and protect recurring revenue.

---

## 18. Key Outputs

The project generated the following outputs:

### Tables

```text
client_missing_summary.csv
record_missing_summary.csv
eda_key_metrics_by_churn.csv
eda_business_insights.csv
network_quality_by_churn.csv
customer_care_by_churn.csv
model_comparison.csv
random_forest_feature_importance.csv
risk_segment_summary.csv
estimated_retention_impact.csv
risk_adjusted_retention_impact.csv
retention_action_plan.csv
final_segment_strategy.csv
```

### Charts

```text
01_churn_distribution.png
02_avg_monthly_usage_by_churn.png
03_change_mou_by_churn.png
04_equipment_age_by_churn.png
05_avg_monthly_revenue_by_churn.png
06_dropped_blocked_calls_by_churn.png
07_customer_care_calls_by_churn.png
08_model_comparison.png
09_feature_importance.png
10_retention_impact.png
```

---

## 19. Tools and Libraries Used

The analysis was conducted in Python using the following libraries:

```text
pandas
numpy
matplotlib
scikit-learn
pathlib
```

Main scikit-learn components used:

```text
train_test_split
Pipeline
ColumnTransformer
SimpleImputer
StandardScaler
OneHotEncoder
LogisticRegression
RandomForestClassifier
classification_report
confusion_matrix
roc_auc_score
```

---

## 20. How to Run the Notebook

1. Place the dataset files in the `data/` folder:

```text
data/Client.csv
data/Record.csv
```

2. Open the notebook:

```text
notebook/GCI_Churn_Project.ipynb
```

3. Run all cells from top to bottom.

4. Generated tables will be saved in:

```text
outputs/tables/
```

5. Generated charts will be saved in:

```text
outputs/charts/
```

---

## 21. Limitations

This project is a Proof of Concept and has several limitations:

* Some columns have unclear business definitions.
* Some demographic columns contain high missing values.
* The dataset is anonymized, so industry-specific assumptions are limited.
* The financial impact calculation is based on estimated retention improvement scenarios.
* The model should be validated with future unseen customer data before full production use.

---

## 22. Next Steps

Recommended next steps for Company A:

1. Validate the model on newer customer data.
2. Run a pilot retention campaign on the High Risk segment.
3. Compare campaign results against a control group.
4. Measure actual retained revenue.
5. Retrain the model periodically.
6. Integrate churn scoring into CRM and marketing systems.

---

## 23. Conclusion

The analysis showed that Company A has a significant churn problem, with 49.56% of customers churning in the observed prediction window. The Random Forest model achieved a ROC-AUC score of 0.6696 and a recall score of 0.6764, outperforming the Logistic Regression baseline.

The model identified equipment age, customer tenure, and usage decline as the strongest churn predictors. By targeting the High Risk segment, where the actual churn rate reached 79.27%, Company A can focus retention resources more efficiently.

The proposed churn risk scoring system can help Company A protect approximately 996K in annual revenue if it achieves a 10% retention improvement among predicted high-risk churners.
