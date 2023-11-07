import pyreadstat
import pandas as pd

data, meta = pyreadstat.read_sav("exp1.sav")

df = pd.DataFrame(data)

print(df.head())
print(df.columns)

# Line 20 is the Shapiro-Wilks test code using scipy.stats package.
from scipy.stats import shapiro
import matplotlib.pyplot as plt

# check normality of various variables.
# assuming victim refers to the money assigned to victims
# assuming PL refers to the money assigned to poland
# assuming global refers to the money assigned to global institutions

columns_to_check = ['guilt', 'shame', 'moral', 'victims', 'PL', 'global']
conditions = df['cond'].unique()

for condition in conditions:
    print(f"\nCondition {int(condition)}:")

    plt.figure(figsize=(15, 12))

    for i, column in enumerate(columns_to_check):
        subset = df[df['cond'] == condition][column]
        stat, p = shapiro(subset)

        if p < 0.001:
            p_value_str = "< .001"
        else:
            p_value_str = f"= {p:.3f}"

        print(f"  Column: {column}, Statistic (W): {stat}, p-value: {p_value_str}")

        # Plotting the distribution in subplots
        ax = plt.subplot(2, 3, i + 1)

        if p > 0.05:
            color = 'green'
            print(f"    - {column} appears to be normally distributed (assumption met).")
        else:
            color = 'red'
            print(f"    - {column} does not appear to be normally distributed (assumption violated).")

        plt.hist(subset, bins=30, color=color, alpha=0.7, density=True, zorder=2)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.7, zorder=1)

        # Adding distribution curve
        from scipy.stats import norm
        import numpy as np

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, subset.mean(), subset.std())
        plt.plot(x, p, 'k', linewidth=2)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust the spacing between subplots
    plt.suptitle(f"Condition {int(condition)}", fontsize=16)
    plt.show()

# code below to run homogeneity of variances.
from scipy.stats import levene

# List of columns to check for homogeneity of variances
columns_to_check = ['guilt', 'shame', 'victims', 'PL', 'global']

for column in columns_to_check:
    # Perform the Levene test
    group1 = df[df['cond'] == 1][column]  # EU
    group2 = df[df['cond'] == 2][column]  # Polish (assuming that is how they labeled it)
    stat, p = levene(group1, group2)

    # format p value (refer to line 24)
    if p < 0.001:
        p_value_str = "< .001"
    else:
        p_value_str = f"= {p:.3f}"

    print(f"Column: {column}, Levene Statistic: {stat}, p-value {p_value_str}")

    # check if variances are equal
    if p > 0.05:
        print(f"  - Variances for {column} appear to be equal across conditions.")
    else:
        print(f"  - Variances for {column} are not equal across conditions.")


