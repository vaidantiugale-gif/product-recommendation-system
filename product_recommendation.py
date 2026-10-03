# Step 12 - Data Cleaning and Leakage Removal
import numpy as np
import os
import pandas as pd

# Dataset path
csv_path = os.path.join(
    "retail_ai_recommendation_personalization_250k.csv"
)

# Load dataset
df = pd.read_csv(csv_path)

print("Original shape:", df.shape)

# --------------------------------------------------
# 1. Remove leakage / post-recommendation columns
# --------------------------------------------------

leakage_columns = [
    "clicked_recommendation",
    "added_to_cart",
    "quantity",
    "purchase_amount"
]

df = df.drop(columns=leakage_columns)

# --------------------------------------------------
# 2. Handle missing loyalty status
# --------------------------------------------------

df["loyalty_status"] = df["loyalty_status"].fillna("Unknown")

# --------------------------------------------------
# 3. Convert interaction date
# --------------------------------------------------

df["interaction_date"] = pd.to_datetime(
    df["interaction_date"],
    errors="coerce"
)

# --------------------------------------------------
# 4. Create date-based features
# --------------------------------------------------

df["month"] = df["interaction_date"].dt.month

df["dayofweek"] = df["interaction_date"].dt.dayofweek

# --------------------------------------------------
# 5. Remove original date column
# --------------------------------------------------

df = df.drop(columns=["interaction_date"])

print("\nData cleaning completed.")

print("New shape:", df.shape)

print("\nRemaining missing values:")

missing = df.isnull().sum()

print(missing[missing > 0])

print("\nRemaining columns:")

print(df.columns.tolist())
# Step 13 - Encode Categorical Features

from sklearn.preprocessing import LabelEncoder

categorical_columns = [
    "customer_segment",
    "loyalty_status",
    "country",
    "device_type",
    "traffic_source",
    "product_category",
    "brand",
    "recommendation_type",
    "payment_method",
    "time_of_day",
    "season"
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()

    df[column + "_enc"] = encoder.fit_transform(
        df[column].astype(str)
    )

    encoders[column] = encoder

print("\nCategorical encoding completed.")

print("\nEncoded columns:")

for column in categorical_columns:
    print(
        column,
        "->",
        column + "_enc"
    )

print("\nFinal shape after encoding:", df.shape)
# Step 14 - Customer Behavioral Features

user_stats = df.groupby("customer_id").agg(
    n_interactions=("product_id", "count"),
    avg_price=("final_price", "mean"),
    avg_discount=("discount_percentage", "mean"),
    avg_session=("session_duration_minutes", "mean"),
    avg_pages=("pages_viewed", "mean"),
    avg_past_views=("past_views", "mean"),
    past_purchases=("past_purchases", "mean"),
    avg_order_value=("avg_order_value", "mean")
).fillna(0)

print("\nCustomer feature engineering completed.")

print("Number of customers:", len(user_stats))

print("Customer feature shape:", user_stats.shape)

print("\nFirst 5 customers:")

print(user_stats.head())
# Step 15 - Customer Segmentation using K-Means

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Features used for customer segmentation
cluster_features = [
    "n_interactions",
    "avg_price",
    "avg_discount",
    "avg_session",
    "avg_pages",
    "avg_past_views",
    "past_purchases",
    "avg_order_value"
]

# Standardize features
scaler = StandardScaler()

X_users = scaler.fit_transform(
    user_stats[cluster_features]
)

# Test different numbers of clusters
inertia = []

for k in range(2, 7):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_users)

    inertia.append(kmeans.inertia_)

# Elbow plot
plt.figure(figsize=(7, 4))

plt.plot(
    range(2, 7),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method - Customer Segmentation")
plt.show()

# Create 4 customer segments
K = 4

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

user_stats["cluster"] = kmeans.fit_predict(X_users)

# Calculate silhouette score using only 5,000 customers
sample_size = min(5000, len(X_users))

random_state = np.random.RandomState(42)

sample_indices = random_state.choice(
    len(X_users),
    size=sample_size,
    replace=False
)

sil_score = silhouette_score(
    X_users[sample_indices],
    user_stats["cluster"].iloc[sample_indices]
)

print("\nSilhouette Score (sample):", round(sil_score, 4))

print("\nCustomer Segment Summary:")

print(
    user_stats
    .groupby("cluster")[cluster_features]
    .mean()
    .round(2)
)
# Step 16 - Prepare ML Dataset

# Add customer cluster to the main dataset
df = df.merge(
    user_stats[["cluster"]],
    left_on="customer_id",
    right_index=True,
    how="left"
)

# Features used by the ML model
feature_columns = [
    "customer_age",
    "product_price",
    "discount_percentage",
    "final_price",
    "recommendation_position",
    "recommendation_score",
    "session_duration_minutes",
    "pages_viewed",
    "search_queries",
    "past_views",
    "past_purchases",
    "avg_order_value",
    "month",
    "dayofweek",
    "cluster"
]

# Add encoded categorical features
feature_columns += [
    column + "_enc"
    for column in categorical_columns
]

# Create X and y
X = df[feature_columns].fillna(0)

y = df["purchased_after_recommendation"]

print("ML dataset prepared.")

print("\nX shape:", X.shape)

print("y shape:", y.shape)

print("\nNumber of features:", len(feature_columns))

print("\nTarget distribution:")
print(y.value_counts())
# Step 17 - Train/Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split completed.")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("Training features:", X_train.shape[1])
print("Testing features:", X_test.shape[1])
# Step 18 - Logistic Regression Model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train, y_train)

# Predictions
y_pred = logistic_model.predict(X_test)

# Probability of purchase
y_prob = logistic_model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nLogistic Regression Results")
print("---------------------------")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(auc, 4))
# Step 19 - Random Forest Model

from sklearn.ensemble import RandomForestClassifier

print("\nTraining Random Forest...")

random_forest_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    n_jobs=2
)

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)
rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)

print("\nRandom Forest Results")
print("--------------------")
print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1 Score :", round(rf_f1, 4))
print("ROC-AUC  :", round(rf_auc, 4))

# Step 20 - Model Comparison

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        rf_accuracy
    ],
    "Precision": [
        precision,
        rf_precision
    ],
    "Recall": [
        recall,
        rf_recall
    ],
    "F1 Score": [
        f1,
        rf_f1
    ],
    "ROC-AUC": [
        auc,
        rf_auc
    ]
})

print("\nModel Comparison")
print("================")
print(results.round(4))
# Step 21 - Random Forest Feature Importance

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")
print("=========================")
print(feature_importance.head(10).round(4))
import matplotlib.pyplot as plt

top_features = feature_importance.head(10)

plt.figure(figsize=(8, 5))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Features - Random Forest")

plt.tight_layout()
plt.show()

# Step 22 - Product Recommendation Function

def recommend_products(customer_id, top_n=5):

    customer_data = df[df["customer_id"] == customer_id].copy()

    if customer_data.empty:
        print("Customer ID not found.")
        return

    # Prepare features
    customer_X = customer_data[feature_columns].fillna(0)

    # Predict purchase probability
    customer_data["purchase_probability"] = (
        random_forest_model.predict_proba(customer_X)[:, 1]
    )

    # Sort products by predicted probability
    recommendations = customer_data.sort_values(
        by="purchase_probability",
        ascending=False
    ).drop_duplicates(
        subset=["product_id"]
    ).head(top_n)

    print("\nRecommended Products")
    print("====================")

    print(
        recommendations[
            [
                "product_id",
                "product_category",
                "brand",
                "product_price",
                "discount_percentage",
                "purchase_probability"
            ]
        ].to_string(index=False)
    )


# Test recommendation
sample_customer = df["customer_id"].iloc[0]

print("\nTesting recommendation for customer:", sample_customer)

recommend_products(
    sample_customer,
    top_n=5
)
# Step 22 - Test Product Recommendations

def recommend_products(customer_id, top_n=5):

    customer_data = df[
        df["customer_id"] == customer_id
    ].copy()

    if customer_data.empty:
        print("Customer ID not found.")
        return

    customer_X = customer_data[
        feature_columns
    ].fillna(0)

    customer_data["purchase_probability"] = (
        random_forest_model
        .predict_proba(customer_X)[:, 1]
    )

    recommendations = (
        customer_data
        .sort_values(
            by="purchase_probability",
            ascending=False
        )
        .drop_duplicates(
            subset=["product_id"]
        )
        .head(top_n)
    )

    print("\nRecommended Products")
    print("====================")

    print(
        recommendations[
            [
                "product_id",
                "product_category",
                "brand",
                "product_price",
                "discount_percentage",
                "purchase_probability"
            ]
        ].to_string(index=False)
    )


# Select a customer from the dataset
sample_customer = df["customer_id"].iloc[0]

print(
    "\nTesting recommendation for customer:",
    sample_customer
)

recommend_products(
    sample_customer,
    top_n=5
)
# Step 23 & 24 - Top-N Evaluation + Hybrid Recommendation

import numpy as np
import pandas as pd


# --------------------------------------------------
# STEP 23 - TOP-N RECOMMENDATION EVALUATION
# --------------------------------------------------

print("\n" + "=" * 50)
print("STEP 23 - TOP-N RECOMMENDATION EVALUATION")
print("=" * 50)

# Get Random Forest probabilities for test data
test_results = df.loc[X_test.index].copy()

test_results["rf_probability"] = rf_prob

# Recommend top 5 products for each customer
top_n = 5

top_recommendations = (
    test_results
    .sort_values(
        ["customer_id", "rf_probability"],
        ascending=[True, False]
    )
    .groupby("customer_id")
    .head(top_n)
)

# Calculate whether at least one recommended product was purchased
customer_hits = (
    top_recommendations
    .groupby("customer_id")["purchased_after_recommendation"]
    .max()
)

hit_rate = customer_hits.mean()

# Precision@5
precision_at_5 = (
    top_recommendations["purchased_after_recommendation"].sum()
    / len(top_recommendations)
)

print("\nTop-N Evaluation Results")
print("------------------------")
print("Top-N :", top_n)
print("Customers evaluated:", len(customer_hits))
print("Hit Rate@5 :", round(hit_rate, 4))
print("Precision@5:", round(precision_at_5, 4))


# --------------------------------------------------
# STEP 24 - HYBRID RECOMMENDATION
# --------------------------------------------------

print("\n" + "=" * 50)
print("STEP 24 - HYBRID RECOMMENDATION")
print("=" * 50)

# Normalize recommendation score
score_min = test_results["recommendation_score"].min()
score_max = test_results["recommendation_score"].max()

if score_max > score_min:
    test_results["normalized_recommendation_score"] = (
        (test_results["recommendation_score"] - score_min)
        / (score_max - score_min)
    )
else:
    test_results["normalized_recommendation_score"] = 0


# Combine Random Forest probability
# with existing recommendation score

MODEL_WEIGHT = 0.7
RECOMMENDER_WEIGHT = 0.3

test_results["hybrid_score"] = (
    MODEL_WEIGHT * test_results["rf_probability"]
    + RECOMMENDER_WEIGHT
    * test_results["normalized_recommendation_score"]
)


# Get Top-5 hybrid recommendations
hybrid_recommendations = (
    test_results
    .sort_values(
        ["customer_id", "hybrid_score"],
        ascending=[True, False]
    )
    .groupby("customer_id")
    .head(top_n)
)


# Evaluate Hybrid Recommendations

hybrid_hits = (
    hybrid_recommendations
    .groupby("customer_id")["purchased_after_recommendation"]
    .max()
)

hybrid_hit_rate = hybrid_hits.mean()

hybrid_precision_at_5 = (
    hybrid_recommendations["purchased_after_recommendation"].sum()
    / len(hybrid_recommendations)
)


print("\nHybrid Recommendation Results")
print("-----------------------------")
print("Model Weight        :", MODEL_WEIGHT)
print("Recommendation Weight:", RECOMMENDER_WEIGHT)
print("Top-N               :", top_n)
print("Customers evaluated :", len(hybrid_hits))
print("Hybrid Hit Rate@5   :", round(hybrid_hit_rate, 4))
print("Hybrid Precision@5  :", round(hybrid_precision_at_5, 4))


# --------------------------------------------------
# SHOW SAMPLE HYBRID RECOMMENDATIONS
# --------------------------------------------------

sample_customer = test_results["customer_id"].iloc[0]

sample_recommendations = (
    hybrid_recommendations[
        hybrid_recommendations["customer_id"] == sample_customer
    ]
)

print("\nSample Customer:", sample_customer)

print("\nTop 5 Hybrid Recommendations")
print("-----------------------------")

print(
    sample_recommendations[
        [
            "product_id",
            "product_category",
            "brand",
            "product_price",
            "discount_percentage",
            "rf_probability",
            "recommendation_score",
            "hybrid_score"
        ]
    ].to_string(index=False)
)
# Step 25 - Save Trained Models and Encoders

import joblib
import os

print("\n" + "=" * 50)
print("STEP 25 - SAVING TRAINED MODELS")
print("=" * 50)

# Create model folder
os.makedirs("models", exist_ok=True)

# Save Random Forest model
joblib.dump(
    random_forest_model,
    "models/random_forest_model.pkl"
)

# Save Logistic Regression model
joblib.dump(
    logistic_model,
    "models/logistic_regression_model.pkl"
)

# Save customer scaler
joblib.dump(
    scaler,
    "models/customer_scaler.pkl"
)

# Save K-Means model
joblib.dump(
    kmeans,
    "models/customer_kmeans.pkl"
)

# Save categorical encoders
joblib.dump(
    encoders,
    "models/categorical_encoders.pkl"
)

# Save feature columns
joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print("\nModels saved successfully!")

print("\nSaved files:")
print("1. random_forest_model.pkl")
print("2. logistic_regression_model.pkl")
print("3. customer_scaler.pkl")
print("4. customer_kmeans.pkl")
print("5. categorical_encoders.pkl")
print("6. feature_columns.pkl")