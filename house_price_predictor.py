import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 特徵與目標
features = ['行政區', '土地面積', '建物總面積', '屋齡', '樓層', '總樓層', '房數', '廳數', '衛數', '電梯']
target = '總價'
X = df[features]
y = df[target]

# 分類與數值欄位
categorical_features = ['行政區']
numerical_features = [col for col in features if col not in categorical_features]

# 建立前處理與模型管線
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("均方誤差 (MSE):", mse)
print("R² 分數:", r2)
