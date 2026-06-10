import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = os.path.join(os.path.dirname(__file__), "Superstore.csv")
df = pd.read_csv(file_path, encoding="latin1")

print("Loading Superstore Dataset...")

print("\nFirst 5 Rows:")
print(df.head())

# Convert Order Date
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Monthly Sales Trend
monthly_sales = (
    df.groupby(df["Order Date"].dt.to_period("M"))["Sales"]
    .sum()
)

plt.figure(figsize=(10,5))
monthly_sales.plot()

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("monthly_sales.png")
plt.close()

# Top 10 Products
top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
top_products.plot(kind="bar")

plt.title("Top 10 Selling Products")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("top_products.png")
plt.close()

# Profit by Category
profit = (
    df.groupby("Category")["Profit"]
    .sum()
)

plt.figure(figsize=(8,5))
profit.plot(kind="bar")

plt.title("Profit by Category")
plt.ylabel("Profit")

plt.tight_layout()
plt.savefig("profit_analysis.png")
plt.close()

print("\nAnalysis Completed Successfully!")
print("\nTop Products:")
print(top_products)

print("\nProfit by Category:")
print(profit)