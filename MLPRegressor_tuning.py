import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import math

# 1️⃣ Load dữ liệu
df = pd.read_csv("../resources/BostonHousing.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 2️⃣ Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Định nghĩa lưới tham số để tuning
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01],
    'solver': ['adam', 'lbfgs'],
    'learning_rate_init': [0.001, 0.01]
}

# 4️⃣ Tạo mô hình cơ bản
mlp = MLPRegressor(max_iter=500, random_state=42)

# 5️⃣ Dùng GridSearchCV để dò tham số tốt nhất
grid = GridSearchCV(estimator=mlp,
                    param_grid=param_grid,
                    scoring='r2',    # có thể đổi thành 'neg_mean_squared_error'
                    cv=5,
                    n_jobs=-1,
                    verbose=2)

grid.fit(X_train, y_train)

# 6️⃣ In kết quả tuning
print("\n===== KẾT QUẢ TUNING =====")
print("Best Params:", grid.best_params_)
print(f"Best Cross-Validation R²: {grid.best_score_:.4f}")

# 7️⃣ Đánh giá trên tập test
best_mlp = grid.best_estimator_
y_pred = best_mlp.predict(X_test)

rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")