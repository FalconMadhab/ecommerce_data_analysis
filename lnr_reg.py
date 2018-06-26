import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Ecommerce Customers")

#Exploratory Data Analysis

sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# Comparision of Time on Website and Yearly Amount Spent.

sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
plt.show()   #Figure_1

# Comparison of Time on App and yearly Amount Spent.

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
plt.show()   #Figure_2

#2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
plt.show()   #Figure_3

#establishing types of relationships across the entire data set.
sns.pairplot(customers)
plt.show()   #Figure_4

#Creating a linear model plot (using seaborn's ) of Yearly Amount Spent vs. Length of Membership.
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
plt.show()   #Figure_5

#Training and Testing Data
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#Training The Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

#Predicting Test Data
predictions=lm.predict(X_test)
plt.scatter(y_test,predictions) 
plt.xlabel('Y Test')
plt.ylabel('Predicte Y')
plt.show()  #Figure_6

#Evaluating the model
#evaluating the model performance by calculating the residual sum of squares and the explained variance score (R^2)
from sklearn import metrics

print('MAE: ',metrics.mean_absolute_error(y_test,predictions))
print('MSE: ',metrics.mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,predictions)))

#Residuals
sns.distplot((y_test-predictions,),bins=50)
plt.show() #Figure_7

#Conclusion Drawn
'''
do we focus our effort on mobile app or website development? 
Or maybe that doesn't even really matter, and Membership Time is what is really important.

'''
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)

#Interpretation

'''
Interpreting the coefficients:

   Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
   Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
   Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
   Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

'''