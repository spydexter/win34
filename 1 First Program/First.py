import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

dataset=pd.read_csv("Advertising.csv")
x=dataset[['TV']]
y=dataset['Sales']



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

from sklearn.linear_model import LinearRegression
slr=LinearRegression()
slr.fit(x_train,y_train)


print('intercept:',slr.intercept_)
print('coefficent:',slr.coef_)
y_pred_slr=slr.predict(x_test)

print('predictionfor test set:{}'.format(y_pred_slr))

slr_diff=pd.DataFrame({'actual value':y_test,'predicted values':y_pred_slr})
slr_diff.head()
plt.scatter(x_test,y_test)

plt.plot(x_test,y_pred_slr,'red')
plt.show()


meanAbErr=metrics.mean_absolute_error(y_test,y_pred_slr)

meanSqErr=metrics.mean_squared_error(y_test,y_pred_slr)

rootmeanSqErr=np.sqrt(metrics.mean_squared_error(y_test,y_pred_slr))

print("Mean Absolute Error:",meanAbErr)
print("Mean Square Error:",meanSqErr)
print("Root Mean Square Error:",rootmeanSqErr)
print('R Squared:{:.2f}'.format(slr.score(x,y)*100))
