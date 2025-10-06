import numpy as np
import matplotlib.pyplot as plt

# 🍎 예제 데이터: 키(cm)와 신발 사이즈(mm)
# 실제 데이터와 유사하게 어느 정도 선형 관계를 가지도록 설정
data = np.array([
    [160, 235],
    [165, 240],
    [170, 250],
    [175, 255],
    [180, 260],
    [185, 270],
    [190, 280]
])

# 키(독립 변수, X)와 신발 사이즈(종속 변수, Y) 분리
X = data[:, 0]  # 키
Y = data[:, 1]  # 신발 사이즈

# 데이터 산점도 시각화
"""
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Data Points')
plt.title('Height vs. Shoe Size')
plt.xlabel('Height (cm)')
plt.ylabel('Shoe Size (mm)')
plt.grid(True)
plt.legend()
plt.show()

print(f"키(X) 데이터: {X}")
print(f"신발 사이즈(Y) 데이터: {Y}")
"""



# 📈 사람이 직접 a와 b를 바꿔가며 최적의 직선 찾기

# 🚨 학습자가 직접 수정할 값 🚨
# 초깃값 (임의의 값)
a = 1.45  # 기울기 (대략 '신발 사이즈 증가량 / 키 증가량')
b = 0 # 절편 (키가 0일 때의 신발 사이즈, 해석은 중요하지 않음)

# -----------------------------------------------------

# 예측 함수: Y = aX + b
Y_pred = a * X + b

# 손실 함수 (Loss Function): 평균 제곱 오차 (Mean Squared Error, MSE) 계산
# (예측값 - 실제값)의 제곱의 평균
mse = np.mean((Y_pred - Y)**2)

# 결과 출력
print(f"💡 현재 설정된 기울기 (a): {a}")
print(f"💡 현재 설정된 절편 (b): {b}")
print(f"⭐ 현재 MSE (손실) 값: {mse:,.2f}") # MSE 값은 작을수록 좋은 모델

# 직선 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', linestyle='-', label=f'Predicted Line: Y = {a:.2f}X + {b:.2f}')
plt.title(f'Finding the Best Line (MSE: {mse:,.2f})')
plt.xlabel('Height (cm)')
plt.ylabel('Shoe Size (mm)')

# 잔차(오차) 시각화 (선택 사항)
# for i in range(len(X)):
#     plt.plot([X[i], X[i]], [Y[i], Y_pred[i]], 'g--', alpha=0.5)

plt.grid(True)
plt.legend()
plt.show()

# ✅ 최적의 값 (참고용)
# a ≈ 2.0, b ≈ -80.0 일 때 MSE가 가장 작아집니다.
# a = 2.0, b = -80.0 일 경우 MSE는 0.00