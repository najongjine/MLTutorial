import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 📊 모든 데이터를 하나의 DataFrame으로 통합
female_data = [
    [160, 230], [162, 235], [165, 240],
    [168, 245], [170, 250], [158, 230],
    [163, 235], [172, 250]
]
male_data = [
    [170, 255], [175, 260], [178, 265],
    [180, 270], [185, 275], [188, 280],
    [173, 255], [182, 270]
]

# 키와 신발 사이즈 데이터만 통합
all_data_numeric = female_data + male_data

# 1. 🐼 Pandas DataFrame 생성
df_single = pd.DataFrame(all_data_numeric, columns=['키(cm)', '신발 사이즈(mm)'])

# 2. 🧠 단순 선형 회귀 모델 학습을 위한 X와 Y 분리
# 성별을 무시하고, '키(cm)' 만을 독립 변수로 사용 (하나의 변수)
X_single = df_single[['키(cm)']]  # 2차원 DataFrame (키)
Y_single = df_single['신발 사이즈(mm)'] # 1차원 Series (신발 사이즈)

single_model = LinearRegression()
single_model.fit(X_single, Y_single)

# 3. 📉 예측: 학습된 모델을 이용해 X 값에 대한 예측 Y 값 생성
Y_pred_single = single_model.predict(X_single)


# 4. 📊 Visualization of Results (Single Line Regression)
plt.figure(figsize=(10, 7))

# --- Actual Data (Scatter Plot) ---
# Display all data points regardless of gender.
plt.scatter(df_single['키(cm)'], df_single['신발 사이즈(mm)'],
            color='gray', label='Actual Data (All Genders)', marker='o', s=80, alpha=0.6)

# --- Linear Regression Line (Single Line) ---
# Draw the single best-fit line representing all points.
plt.plot(df_single['키(cm)'], Y_pred_single,
         color='green', label='Linear Regression Prediction Line (Single)', linewidth=3)

# Set graph title and axis labels
plt.title('👟 Simple Linear Regression Analysis of Shoe Size by Height (One Prediction Line)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Height (cm)', fontsize=14)
plt.ylabel('Shoe Size (mm)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()

# 🔍 모델 결과 확인
print("\n--- 단순 선형 회귀 모델 결과 ---")
print(f"기울기 (Coefficient): {single_model.coef_[0]:.4f}")
print(f"절편 (Intercept): {single_model.intercept_:.4f}")
print(f"회귀 방정식: 신발 사이즈(Y) = {single_model.coef_[0]:.4f} * 키(X) + {single_model.intercept_:.4f}")