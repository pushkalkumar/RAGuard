import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load model using joblib instead of pickle
model = joblib.load("results/final/patch_classifier_model.pkl")

# Get feature importances (works for tree-based or linear models)
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
elif hasattr(model, "coef_"):
    importances = np.abs(model.coef_).flatten()
else:
    raise ValueError("Model does not expose feature importances")

# Generate mock feature names (replace with actual ones if available)
feature_names = [f"Feature_{i}" for i in range(len(importances))]
top_idx = np.argsort(importances)[-10:]  # top 10 features
top_features = np.array(feature_names)[top_idx]
top_values = importances[top_idx]

plt.figure(figsize=(7, 5))
plt.barh(top_features, top_values, color="#1f77b4", alpha=0.8)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Top Feature Importances in ZK Patch Classifier", fontsize=13, pad=10)
plt.tight_layout()
plt.savefig("results/final/patch_feature_importance.png", dpi=600, bbox_inches="tight")
plt.show()

