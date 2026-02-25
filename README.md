# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the car price dataset, select relevant numerical features (enginesize, horsepower, citympg, highwaympg) as input variables, and set price as the target variable. Split the data into training and testing sets.
2. Apply standardization to the training features using StandardScaler and transform the testing features using the same scaler to ensure consistent feature scaling.
3. Train a Linear Regression model using the scaled training data, predict prices for the test data, and evaluate model performance using MSE, RMSE, and R-squared metrics along with model coefficients.
4.Check linearity using actual vs predicted plots, test independence of errors using the Durbin–Watson statistic, assess homoscedasticity through residual plots, and verify normality of residuals using histogram and Q–Q plots. 

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

x=df[["enginesize","horsepower","citympg","highwaympg"]]
y=df["price"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)
print('Name: SARANYA R')
print('Reg NO: 212225040384')
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature:}: {coef:}")
print(f"Intercept: {model.intercept_:}") 

print("\nMODEL PERFORMANCE:")
print(f"MSE: {mean_squared_error(y_test,y_pred):}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test,y_pred)):}")
print(f"R-squared: {r2_score(y_test,y_pred):}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Price")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-watson statistic:{dw_test:.2f}",
      "\n(values close to 2 indicate no autocorrelation)")
      
plt.figure(figsize=(10, 5)) 
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted") 
plt.xlabel("Predicted Price ($)") 
plt.ylabel("Residuals ($)")
plt.grid(True) 
plt.show() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) 
sns.histplot(residuals, kde=True, ax=ax1) 
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()

```

## Output:
<img width="634" height="349" alt="image" src="https://github.com/user-attachments/assets/ed5d682e-cc65-46e0-a7dc-c48b210ee70a" />
<img width="1498" height="657" alt="Screenshot 2026-02-04 092426" src="https://github.com/user-attachments/assets/45062759-5980-465f-b359-7492ba0d9d73" />
<img width="1490" height="654" alt="Screenshot 2026-02-04 092452" src="https://github.com/user-attachments/assets/c419df3d-f87d-48bf-8f9a-64ec7b8d5593" />
<img width="1524" height="577" alt="image" src="https://github.com/user-attachments/assets/c13090fa-fa37-4ff6-a9c6-bdcf0b4bf110" />

## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
