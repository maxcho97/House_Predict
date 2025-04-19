# House_Predict
# Taipei House Price Predictor 

這是一個使用台北市房價資料建立的機器學習預測模型專案，能夠根據房屋條件預測房價。

---

## 專案內容

- `Taipei_house.csv`：原始資料，包含行政區、建物面積、屋齡、房數等資訊。
- `house_price_predictor.py`：房價預測模型程式，使用線性回歸建立預測。
- `README.md`：本說明文件。

---

## 使用技術

- Python 3
- pandas / scikit-learn
- Linear Regression（線性回歸）
- OneHotEncoder + Pipeline 處理分類資料

---

## 如何使用

### 1. 安裝依賴套件
```bash
pip install pandas scikit-learn
```

### 2. 確保資料檔與程式在同一目錄中
- `Taipei_house.csv`
- `house_price_predictor.py`

### 3. 執行程式
```bash
python house_price_predictor.py
```

### 4. 結果輸出
程式會印出：
- 均方誤差（MSE）
- R² 模型分數



