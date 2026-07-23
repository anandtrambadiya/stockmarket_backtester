from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd

def train_model(df):
    feature_cols = ['SMA20', 'SMA50', 'RSI_14', 'BB_upper', 'BB_lower', 'Volatility_20']
    X = df[feature_cols]
    y = df['Target']
    

    # spliting feature/input data(col)
    split = int(len(X)*0.8)
    
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    

    #train on random forest
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"Precision: {precision_score(y_test, predictions)}")

    print(f"recall: {recall_score(y_test, predictions)}")
    print(f"roc: {roc_auc_score(y_test, predictions)}")

    print()
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    print(importance.sort_values(ascending=False))

    return model, predictions, X_test, y_test, accuracy_score(y_test, predictions)
