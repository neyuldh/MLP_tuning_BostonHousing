import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1️⃣ Load dữ liệu
df = pd.read_csv("../resources/BostonHousing.csv")

X = df.iloc[:, :-1]
y_continuous = df.iloc[:, -1]

# Biến bài toán regression thành classification
# Nếu giá nhà >= median → 1 (High), ngược lại → 0 (Low)
threshold = np.median(y_continuous)
y = (y_continuous >= threshold).astype(int)

# 2️⃣ Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4️⃣ Lưới tham số tuning
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01],
    'solver': ['adam', 'lbfgs'],
    'learning_rate_init': [0.001, 0.01]
}

# 5️⃣ Tạo model
mlp = MLPClassifier(max_iter=500, random_state=42)

# 6️⃣ Tuning tham số bằng GridSearchCV
grid = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

# 7️⃣ In kết quả
print("\n===== KẾT QUẢ TUNING =====")
print("Best Params:", grid.best_params_)
print(f"Best CV Accuracy: {grid.best_score_:.4f}")

# 8️⃣ Đánh giá trên test set
best_mlp = grid.best_estimator_
y_pred = best_mlp.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on Test Set: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))