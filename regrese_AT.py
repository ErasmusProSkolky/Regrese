import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn
import matplotlib.pyplot as plt
import numpy as np

#vykresleni grafu zavislosti obsazenosti skolek na porodnosti
df = pd.read_csv("regrese_AT.csv")
df["narozene_deti"] = df["2016"] + df["2017"] + df["2018"] + df["2019"]
X=df[['2016', '2017', '2018', '2019']]
g = seaborn.JointGrid(data=df, x="narozene_deti", y="OBSAZENOST")
g.plot_joint(seaborn.scatterplot, legend=False)
plt.show()

df = pd.read_csv("regrese_AT.csv")

#do osy X vstupuji pocty narozenych deti, ktere v roce 2021 mohou navstevovat skolku
#model vyhodnoti, v jake mire se tyto rocniky podileji na obszenosti
df["total"] = df["2015"] + df["2016"] + df["2017"] + df["2018"] + df["2019"]
print(df.head())
X=df[['2015','2016', '2017', '2018', '2019']]
Y=df["OBSAZENOST"]

model=sm.OLS(Y,X).fit()
print(model.summary())
#lze otestovat i RLM: model=sm.RLM(Y,X).fit()

#test predikce na stavajicich datech
predictions=model.predict(X)
print(predictions)

#predikce pro rok 2023
df1 = pd.read_csv("regrese_AT.csv")
df1["total"]= df1["2017"] + df1["2018"] + df1["2019"] + df1["2020"] + df1["2021"]
X1=df1[['2017','2018', '2019', '2020', '2021']]
predictions=model.predict(X1)
print(predictions)