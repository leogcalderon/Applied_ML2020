# Assignment 1
## Task 1: Density Plots
1. Plot the longitude vs latitude several ways within a single figure (each in its own axes):
  1. Using the matplotlib defaults.
  2. Adjusting alpha and marker size to compensate for overplotting.
  3. Using a hexbin plot.
  4. Subsampling the dataset.

For each but the first one, ensure that all the plotting area is used in a reasonable way and thatas much information as possible is conveyed; this is somewhat subjective and there is no oneright answer.

2. In what areas are most of the anomalies (measurements) located?

## Task 2: Visualizing class membership
Visualize the distribution of Brightness temperature I-4 as a histogram (with appropriatesettings). Let’s assume we are certain of a fire if the value of temperature I-4 is saturated asvisible from the histogram.
1. Do a small multiples plot of whether the brightness is saturated, i.e. do one plot of lat vslong for those points with brightness saturated and a separate for those who are not (within thesame figure on separate axes). You can pick any of the methods from 1.1 that you find mostsuitable. Can you spot differences in the distributions?
2. Plot both groups in the same axes with different colors. Try changing the order of plottingthe two classes (i.e. draw the saturated first then the non-saturated or the other way around).Make sure to include a legend. How does that impact the result?
3. Can you find a better way to compare the two distributions
