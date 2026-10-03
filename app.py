from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

random_forest_model = joblib.load(
    "models/random_forest_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

encoders = joblib.load(
    "models/categorical_encoders.pkl"
)

scaler = joblib.load(
    "models/customer_scaler.pkl"
)

kmeans = joblib.load(
    "models/customer_kmeans.pkl"
)


# --------------------------------------------------
# LOAD RAW DATA
# --------------------------------------------------

df = pd.read_csv(
    "retail_ai_recommendation_personalization_250k.csv"
)


# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------

# Remove leakage columns
leakage_columns = [
    "clicked_recommendation",
    "added_to_cart",
    "quantity",
    "purchase_amount"
]

df = df.drop(
    columns=leakage_columns
)

# Fill missing loyalty status
df["loyalty_status"] = df[
    "loyalty_status"
].fillna("Unknown")

# Convert date
df["interaction_date"] = pd.to_datetime(
    df["interaction_date"],
    errors="coerce"
)

# Create date features
df["month"] = df[
    "interaction_date"
].dt.month

df["dayofweek"] = df[
    "interaction_date"
].dt.dayofweek

# Remove original date
df = df.drop(
    columns=["interaction_date"]
)


# --------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

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

for column in categorical_columns:

    df[column + "_enc"] = encoders[
        column
    ].transform(
        df[column].astype(str)
    )


# --------------------------------------------------
# CUSTOMER FEATURES
# --------------------------------------------------

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

user_stats = df.groupby(
    "customer_id"
).agg(
    n_interactions=("product_id", "count"),
    avg_price=("final_price", "mean"),
    avg_discount=("discount_percentage", "mean"),
    avg_session=("session_duration_minutes", "mean"),
    avg_pages=("pages_viewed", "mean"),
    avg_past_views=("past_views", "mean"),
    past_purchases=("past_purchases", "mean"),
    avg_order_value=("avg_order_value", "mean")
).fillna(0)


# --------------------------------------------------
# CUSTOMER CLUSTER
# --------------------------------------------------

X_users = scaler.transform(
    user_stats[cluster_features]
)

user_stats["cluster"] = kmeans.predict(
    X_users
)


# Add cluster to main dataframe
df = df.merge(
    user_stats[["cluster"]],
    left_on="customer_id",
    right_index=True,
    how="left"
)


print("Data preprocessing completed.")
print("Final dataset shape:", df.shape)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# RECOMMENDATION
# --------------------------------------------------

@app.route(
    "/recommend",
    methods=["POST"]
)
def recommend():

    customer_id = request.form[
        "customer_id"
    ].strip()

    # Find customer
    customer_data = df[
        df["customer_id"] == customer_id
    ].copy()

    if customer_data.empty:

        return render_template(
            "index.html",
            error="Customer ID not found."
        )


    # Prepare ML features
    customer_X = customer_data[
        feature_columns
    ].fillna(0)


    # Predict purchase probability
    customer_data[
        "purchase_probability"
    ] = random_forest_model.predict_proba(
        customer_X
    )[:, 1]


    # Sort recommendations
    recommendations = (
        customer_data
        .sort_values(
            by="purchase_probability",
            ascending=False
        )
        .drop_duplicates(
            subset=["product_id"]
        )
        .head(5)
    )


    # Select columns for website
    recommendations = recommendations[
        [
            "product_id",
            "product_category",
            "brand",
            "product_price",
            "discount_percentage",
            "purchase_probability"
        ]
    ].to_dict("records")


    return render_template(
        "index.html",
        recommendations=recommendations,
        customer_id=customer_id
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )