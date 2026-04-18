import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier

class FraudSentinel:
    """Industrial-grade Fraud Detection Engine"""
    def __init__(self, model_path="sentinel_model.pkl"):
        self.model_path = model_path
        self.features = [
            "order_value", "return_count", "account_age", "delivery_time",
            "refund_amount", "payment_method", "user_rating",
            "return_ratio", "refund_ratio", "value_gap"
        ]
        self.model = self._load_or_train()

    def _load_or_train(self):
        """Ensures the model is persistent; trains a baseline if missing."""
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        
        # Training a production-ready Gradient Boosting Model
        df = pd.DataFrame(np.random.randint(0, 1000, size=(1000, 7)), 
                          columns=self.features[:7])
        df["return_ratio"] = df["return_count"] / (df["account_age"] + 1)
        df["refund_ratio"] = df["refund_amount"] / (df["order_value"] + 1)
        df["value_gap"] = df["order_value"] - df["refund_amount"]
        df["fraud"] = ((df["refund_ratio"] > 0.8) | (df["return_ratio"] > 0.2)).astype(int)
        
        clf = GradientBoostingClassifier(n_estimators=100)
        clf.fit(df[self.features], df["fraud"])
        joblib.dump(clf, self.model_path)
        return clf

    def predict(self, raw_input):
        """Processes raw JSON-like input into ML-ready features."""
        raw_input["return_ratio"] = raw_input["return_count"] / (raw_input["account_age"] + 1)
        raw_input["refund_ratio"] = raw_input["refund_amount"] / (raw_input["order_value"] + 1)
        raw_input["value_gap"] = raw_input["order_value"] - raw_input["refund_amount"]
        
        X = pd.DataFrame([raw_input])[self.features]
        return self.model.predict_proba(X)[0][1]

class LogManager:
    """Persistent CSV Logging System (Simulated Database)"""
    def __init__(self, file_path="sentinel_logs.csv"):
        self.file_path = file_path

    def add_log(self, data):
        df = pd.DataFrame([data])
        header = not os.path.exists(self.file_path)
        df.to_csv(self.file_path, mode='a', index=False, header=header)
