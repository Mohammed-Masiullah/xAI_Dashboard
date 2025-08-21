# save_sample.py
import pandas as pd, joblib

# Load the same features used during training
bundle = joblib.load("models/models.joblib")
X_columns = bundle.get("X_columns")  # if you saved this during training
# Or load your raw dataset with these columns:
df = pd.read_csv("../data/adult.csv")

# Select only the columns used by the model (if available)
if X_columns is not None:
    df = df[X_columns]

# Create a modest background sample for SHAP/LIME (e.g., 500 rows)
sample = df.sample(500, random_state=7)
sample.to_csv("../data/sample_inputs.csv", index=False)
print("Wrote data/sample_inputs.csv")
