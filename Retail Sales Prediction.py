import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# RETAIL STORE DATASET

data = {
    "Month": [
        "Jan", "Feb", "Mar", "Apr", "May",
        "Jun", "Jul", "Aug", "Sep", "Oct"
    ],

    "Advertising_Spend": [
        2000, 2500, 3000, 3500, 4000,
        4200, 4500, 4700, 5000, 5500
    ],

    "Customers": [
        200, 240, 280, 300, 350,
        370, 390, 410, 450, 480
    ],

    "Sales": [
        15000, 18000, 22000, 25000, 30000,
        32000, 35000, 37000, 42000, 45000
    ]
}

df = pd.DataFrame(data)

# DISPLAY DATASET

print("\nRETAIL STORE DATASET\n")
print(df)

# DATA CLEANING

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nDataset Information:\n")
print(df.info())

# DATA ANALYSIS

print("\nAverage Sales:")
print(df["Sales"].mean())

print("\nHighest Sales:")
print(df["Sales"].max())

print("\nTotal Customers:")
print(df["Customers"].sum())

# VISUALIZATION 

plt.figure(figsize=(8,5))

plt.plot(
    df["Month"],df["Sales"],marker='o'
)

plt.title("Monthly Sales Growth")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.grid(True)

plt.show()

# VISUALIZATION 

plt.figure(figsize=(8,5))

plt.bar(
    df["Month"],df["Customers"]
)

plt.title("Customers Per Month")
plt.xlabel("Month")
plt.ylabel("Number of Customers")

plt.show()

# MACHINE LEARNING MODE

X = df[["Advertising_Spend", "Customers"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

# SALES PREDICTION

future_data = [[6000, 520]]

predicted_sales = model.predict(future_data)

print("\nPredicted Future Sales:")
print(predicted_sales[0])

# MODEL EVALUATION

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error:")
print(mse)

