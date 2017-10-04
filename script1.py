import random

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:\ME\scripts\stats-proj-code\Salary_Data.csv")  # change file name
Y_values = data.iloc[:, 1].values
X_values = data.iloc[:, 0].values

while Y_values.size > 15:
    randnum = random.randint(0, Y_values.size - 1)
    Y_values = np.delete(Y_values, randnum)
    X_values = np.delete(X_values, randnum)
# print(Y_values, X_values)

f = open("data_used.csv", "w+")
f.write("X values: " + np.array_str(X_values) + "\n")
f.write("Y values: " + np.array_str(Y_values) + "\n")

slope, intercept, r_value, p_value, std_er = stats.linregress(
    X_values, Y_values)
X_values = np.reshape(X_values, (X_values.size, 1))
Y_values = np.reshape(Y_values, (X_values.size, 1))
regressor = LinearRegression()
regressor.fit(X_values, Y_values)

print("Stats project calculator thing by Phan Kandula")
print("slope: ", slope)
print("intercept: ", intercept)
print("r: ", r_value)
print("r2: ", r_value * r_value)
print("")
f.write("\n")
f.write("Stats project calculator thing by Phan Kandula" + '\n')
f.write("slope: " + str(slope) + '\n')
f.write("intercept: " + str(intercept) + '\n')
f.write("r: " + str(r_value) + '\n')
f.write("r2: " + str(r_value * r_value) + '\n')
f.write('\n')


plt.scatter(X_values, Y_values)
plt.plot(X_values, regressor.predict(X_values), color="red")
plt.text(np.min(X_values), np.max(Y_values),
         r'y =' + str(intercept) + ' + ' + str(slope) + '(x)', fontsize=15, color='red')
plt.title("Linear Regression Model")
plt.xlabel("Experience(Years)")  # change the X axis title
plt.ylabel("Salary(Dollars)")  # change the Y axis title
plt.show()

resid = np.subtract(Y_values, regressor.predict(X_values))
plt.scatter(X_values, resid)
plt.show()

plt.boxplot(X_values, 0, 'gd', 0)
plt.title("Boxplot X values")
plt.xlabel("Experience(years)")  # change the X axis title
plt.show()

plt.boxplot(Y_values, 0, 'gd', 0)
plt.title("Boxplot Y values")
plt.xlabel("Salary(Dollars)")  # change the Y axis title
plt.show()

print("How to use the following stuff: the first column(before the comma) contains values for the X values. The second column containst values for the Y values ")
print("5 number summary:")
print("min:", np.min(X_values), ", ", np.min(Y_values))
print("Q1:", np.percentile(X_values, 25), ", ", np.percentile(Y_values, 25))
print("med:", np.median(X_values), ", ", np.median(Y_values))
print("Q3:", np.percentile(X_values, 75), ", ", np.percentile(Y_values, 75))
print("max:", np.max(X_values), ", ", np.max(Y_values))
print("")
print("to check if normal-")
print("mean: ", np.mean(X_values), ", ", np.mean(Y_values))
print("median: ", np.median(X_values), ", ", np.median(Y_values))
print(
    "To find mode, use the value found after \"mode=array([[\" and the number of times it occurs is written after \" count=array([[ \" ")
print("mode: ", stats.mode(X_values), ", ", stats.mode(Y_values))
print("")
f.write("5 number summary: \n")
f.write("min:" + str(np.min(X_values)) + ", " + str(np.min(Y_values)) + "\n")
f.write("Q1: " + str(np.percentile(X_values, 25)) +
        ", " + str(np.percentile(Y_values, 25)) + '\n')
f.write("median: " + str(np.median(X_values)) +
        ", " + str(np.median(Y_values)) + "\n")
f.write("Q3: " + str(np.percentile(X_values, 75)) +
        ", " + str(np.percentile(Y_values, 75)) + '\n')
f.write("max:" + str(np.max(X_values)) + ", " + str(np.max(Y_values)) + '\n')

f.write('\n')
f.write("to check if normal- \n")
f.write("mean: " + str(np.mean(X_values)) +
        ", " + str(np.mean(Y_values)) + "\n")
f.write("median: " + str(np.median(X_values)) +
        ", " + str(np.median(Y_values)) + "\n")
f.write("mode: " + str(stats.mode(X_values)) +
        ", " + str(stats.mode(Y_values)) + "\n")
f.write('\n')


def z_score_calc(num, arr):

    return (num - np.mean(arr)) / np.std(arr)


z_num = 3  # the value in the dataset for which you want to compute the z score

# change X/Y_values to the axis you want to use in the line below
z_score = z_score_calc(z_num, X_values)
normcdf = norm.cdf(z_score)
print("Z Score:", z_score_calc(3, X_values),
      ",", z_score_calc(83088, Y_values))
print("Percentile(NormCDF): ", norm.cdf(z_score_calc(
    3, X_values)), ", ", norm.cdf(z_score_calc(83088, Y_values)))
print("invNorm- IQR Q1: ", np.percentile(X_values, 25),
      ", ", np.percentile(Y_values, 25))
print("invNorm- IQR Q3: ", np.percentile(X_values, 75),
      ", ", np.percentile(Y_values, 75))
print("IQR: ", (np.percentile(X_values, 75) - np.percentile(X_values, 25)),
      ", ", (np.percentile(Y_values, 75) - np.percentile(Y_values, 25)))

f.write("Z Score:" + str(z_score_calc(3, X_values)) +
        ", " + str(z_score_calc(83088, Y_values)) + '\n')
f.write("Percentile(normcdf):" + str(norm.cdf(z_score_calc(
    3, X_values))) + ", " + str(norm.cdf(83088, Y_values)) + '\n')
f.write("invNorm- IQR Q1: " + str(np.percentile(X_values, 25)) +
        ", " + str(np.percentile(Y_values, 25)) + '\n')
f.write("invNorm- IQR Q3: " + str(np.percentile(X_values, 75)) +
        ", " + str(np.percentile(Y_values, 75)) + '\n')
f.write("IQR: " + str(np.percentile(X_values, 75) - np.percentile(X_values, 25)) +
        ", " + str(np.percentile(Y_values, 75) - np.percentile(Y_values, 25)) + '\n')

ax = plt.subplots()
n, bins, patches = ax.hist(X_values, 10, normed=1)
y = mlab.normpdf(bins, np.mean(X_values), np.std(X_values))
plt.plot(bins, y, '--')
plt.show()





f.close()
