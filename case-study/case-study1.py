import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Veriler
deneyim_yili = np.array([5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1]).reshape(-1, 1)
maas = np.array([600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380])

# Doğrusal regresyon modeli
model = LinearRegression().fit(deneyim_yili, maas)

# Eğim (b) ve kesme terimi (w)
b = model.coef_[0]
w = model.intercept_

# Tahminler
tahminler = model.predict(deneyim_yili)

# Ortalama karesel hata (MSE)
mse = mean_squared_error(maas, tahminler)

print("Ortalama Kare Hata (MSE):", mse)

# RMSE
rmse = np.sqrt(mse)

#MAE
mae = np.mean(np.abs(maas - tahminler))
