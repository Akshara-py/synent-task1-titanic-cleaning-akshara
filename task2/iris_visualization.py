import os
import seaborn as sns
import matplotlib.pyplot as plt

# Create folder for graphs
os.makedirs("plots", exist_ok=True)

# Load Iris dataset
iris = sns.load_dataset("iris")

print(iris.head())

# Histogram
plt.figure(figsize=(8,5))
plt.hist(iris["sepal_length"], bins=15)
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.savefig("plots/histogram.png")
plt.close()

# Scatter Plot
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=iris,
    x="sepal_length",
    y="petal_length",
    hue="species"
)

plt.title("Sepal Length vs Petal Length")
plt.savefig("plots/scatterplot.png")
plt.close()

# Bar Chart
avg = iris.groupby("species")["sepal_length"].mean()

plt.figure(figsize=(8,5))
avg.plot(kind="bar")

plt.title("Average Sepal Length by Species")
plt.ylabel("Sepal Length")

plt.savefig("plots/barchart.png")
plt.close()

print("All plots generated successfully.")