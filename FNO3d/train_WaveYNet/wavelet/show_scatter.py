import csv
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18, "font.family": "serif"})

# Input file paths
input_file = "wavelet_result.csv"

# Read the data from the first CSV file
with open(input_file, "r") as file1:
    reader = csv.reader(file1)
    reader = list(reader)[1:]
    data = {row[0]: row for row in reader}

# Extract the 2nd, 3rd, 4th, and 5th columns for plotting
x1 = [float(data[row][1]) for row in data]
y1 = [float(data[row][2]) for row in data]
x2 = [float(data[row][3]) for row in data]
y2 = [float(data[row][4]) for row in data]

# Create a scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(x1, y1, color="blue", label="First and second term removed")
plt.scatter(x2, y2, color="red", label="First terms removed")
plt.xlabel("MAE")
plt.ylabel("Compression ratio")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Wavelet transform and inverse")
plt.savefig("scatter_plot.png", bbox_inches="tight", dpi=150)
