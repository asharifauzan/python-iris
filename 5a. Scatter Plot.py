import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('iris.csv')

sns.scatterplot(x="sepals-length",
                y="sepals-width",
                hue='label',
                data=data).set_title('sebaran sepals')
plt.figure(1)
plt.show()
