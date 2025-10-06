import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# -------------------- 데이터 생성 및 모델 학습 (이전 코드 유지) --------------------
# 1. 데이터 생성
N = 200
np.random.seed(42)
NOISE = 35.0
age_A = np.random.uniform(1, 22, N).reshape(-1, 1)
age_B = np.random.uniform(1, 22, N).reshape(-1, 1)
specy_A = np.zeros_like(age_A)
specy_B = np.ones_like(age_B)
volume_A = 100 + 0.5 * age_A + 4 * (age_A**2) + np.random.normal(0, NOISE, (N, 1))
volume_B = 150 + 2.0 * age_B + 4 * (age_B**2) + np.random.normal(0, NOISE, (N, 1))

X_features = np.vstack([np.hstack([age_A, specy_A]), np.hstack([age_B, specy_B])])
y_target = np.vstack([volume_A, volume_B]).ravel()
df_data = pd.DataFrame(X_features, columns=['Age_Weeks', 'Specy_Encoded'])
df_data['Volume_cm3'] = y_target

# 2. 모델 학습
model_linear = LinearRegression().fit(X_features, y_target)
r2_linear = model_linear.score(X_features, y_target)

model_poly = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                           LinearRegression()).fit(X_features, y_target)
r2_poly = model_poly.score(X_features, y_target)

# 3. 예측값 계산
age_range = np.linspace(1, 22, 100).reshape(-1, 1)
X_plot_A = np.hstack([age_range, np.zeros_like(age_range)])
X_plot_B = np.hstack([age_range, np.ones_like(age_range)])

y_linear_A = model_linear.predict(X_plot_A)
y_linear_B = model_linear.predict(X_plot_B)
y_poly_A = model_poly.predict(X_plot_A)
y_poly_B = model_poly.predict(X_plot_B)
# ----------------------------------------------------------------------------------

## 1. 일반 선형 모델 (직선)의 실패 📉

# 그래프 1: 일반 선형 회귀 모델 시각화
plt.figure(figsize=(10, 6))
# 원본 데이터 산점도
plt.scatter(df_data['Age_Weeks'], df_data['Volume_cm3'], c=df_data['Specy_Encoded'], cmap='viridis', alpha=0.5, s=20, label='Real Data (Specy A/B)')
# 선형 예측 (직선)
plt.plot(age_range, y_linear_A, 'r--', label=f'Linear A: R²={r2_linear:.2f}', linewidth=3)
plt.plot(age_range, y_linear_B, 'b--', label=f'Linear B: R²={r2_linear:.2f}', linewidth=3)

plt.title('Graph 1. Linear Regression (Degree=1): Cannot Capture Curve')
plt.xlabel('Age (Weeks)')
plt.ylabel('Volume/Weight')
plt.legend()
plt.grid(True)
plt.show()

# ---

## 2. 2차 다항 모델 (곡선)의 성공 🎉

# 그래프 2: 2차 다항 회귀 모델 시각화
plt.figure(figsize=(10, 6))
# 원본 데이터 산점도
plt.scatter(df_data['Age_Weeks'], df_data['Volume_cm3'], c=df_data['Specy_Encoded'], cmap='viridis', alpha=0.5, s=20, label='Real Data (Specy A/B)')
# 다항 예측 (곡선)
plt.plot(age_range, y_poly_A, 'r-', label=f'Poly A: R²={r2_poly:.4f}', linewidth=3)
plt.plot(age_range, y_poly_B, 'b-', label=f'Poly B: R²={r2_poly:.4f}', linewidth=3)

plt.title('Graph 2. Polynomial Regression (Degree=2): Captures Non-linear Growth')
plt.xlabel('Age (Weeks)')
plt.ylabel('Volume/Weight')
plt.legend()
plt.grid(True)
plt.show()

# ---

## 모델 성능 요약

print("=====================================================")
print("             모델 성능 및 분석 결과 요약             ")
print("=====================================================")
print(f"1. 일반 선형 회귀 (직선) 정확도 (R²): {r2_linear:.4f}")
print(f"2. 2차 다항 회귀 (곡선) 정확도 (R²): {r2_poly:.4f}")
print("\n🔥 **핵심 결론:**")
print("- **그래프 1 (직선)**: 실제 데이터가 곡선인데 모델이 직선이라서 오차가 큽니다. (R² 낮음)")
print("- **그래프 2 (곡선)**: PolynomialFeatures 덕분에 곡선을 정확히 예측합니다. (R² 매우 높음)")
print("=====================================================")