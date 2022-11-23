import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("regresetest.csv")
g = seaborn.JointGrid(data=data, x="Geburten", y="betreute_kinder")
g.plot_joint(seaborn.scatterplot, legend=False)
plt.show()

df = pd.read_csv("regresetest.csv")

X=df["Geburten"]
Y=df["betreute_kinder"]

model=sm.OLS(Y,X).fit()
print(model.summary())

predictions=model.predict(X)
print(predictions)

#df1 = pd.read_csv("predikce.csv")
#test_kinder=df1["kinder_2017-2020"]
#predictions=model.predict(test_kinder)
#print(predictions)