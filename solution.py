import pandas as pd  # module used for data processing, CSV file read writes
import seaborn as sns # module used for data visualization
import matplotlib.pyplot as plt # module used for data visualization
import numpy as np # Linear algebra module

# Below are the modules used for linear regression and model training 
# for data prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Used only for slowing down the display in the terminal , otherwise not needed.
from time import sleep
sleep_time = 1

'''
WHY WE ARE DOING THIS:
	The aim of this project is to estimate Weight of the fish indivuduals
	from their measurements through using linear regression model.

   This project can be improved to use in fish farms. 
   Individual fish swimming in front of the camera can be measured from the video image and the Weight 
   of the fish can be estimated through the linear regression model.
'''

df = pd.read_csv("Fish.csv")

# Print the first few rows of the data we read
print(f"Head of the dataframe===>\n{df.head()}",end='\n----------------------------------------\n')

sleep(sleep_time)

# Next we just print out some general information about the dataset we have
print("General information of the dataframe===>")
print(df.info(),end='\n----------------------------------------\n')

sleep(sleep_time)

# In order to do data prediction and analysis we often require no NULL values in the dataset , so in the next step
# We check for None /  NULL values in the dataframe
print("Checking for None values in the dataframe===>")

# We check for Null using the isnull() function , the isnull() function checks if there is a null value and returns a boolean , either a True or a False.
# Next we check t see if the sum of those boolean values is greater than 0 
# If there are values >0 then there exists null values in the dataset
print(df.isnull().sum())
print("There is no null in the data set",end='\n----------------------------------------\n')

sleep(sleep_time)


# Next we check the Species column in the dataset 
print("Print the number of types of each species available in the dataframe===>")
print(df.Species.value_counts())
print("We see some 7 variety of fish species",end='\n----------------------------------------\n')
sns.countplot(data = df ,x = 'Species')
plt.show()

sleep(sleep_time)

print("Next we plot the scatter plots for each parameter with respect to the Weight of the fish")
sns.pairplot(data = df ,x_vars = ['Length1','Length2','Length3','Height','Width'], y_vars = 'Weight', hue = 'Species')
plt.show()

sleep(sleep_time)

print("As we see, our dependent variable-'weight' has linear relationship with all other variables",end='\n----------------------------------------\n')

sleep(sleep_time)

# We also check for correlation between the parameters .
# This is done using a heatmap which has a value depending on how much the data correlates
print("Next let us check for correlation between the parameters using a heatmap")
sns.heatmap(df.corr(), annot = True)
plt.show()
print("We see that there are high correlation between variables going on.",end='\n----------------------------------------\n')


# And to get a bigger picture of the data we plot a big scatter plot between all the columns 
# NOTE:
# 		THIS COULD BE SLOW ON OLD HARDWARE. COMMENT OUT THE BELOW TWO LINES IF YOUR PC IS LAGGING TOO MUCH
sns.pairplot(df, kind='scatter', hue='Species')
plt.show()

sleep(sleep_time)

# Below is one of the important parts of the solution
# In every dataset we get , there will exist a few outliers in the data.
# For better prediction and visualization we often discard those outliers
# But in order to discard them we first need to find them 
# In this case I will use box plots to show the data and find the outliers
# Removal of these erroneous data values can often greatly impact the outcome 
# 	and offer better predictions 

# We take each coloumn by itself and find the outliers

# 1. Weight	

# First plot the box plot
print("Plot a Box Plot for the weight data")
sns.boxplot(x=df['Weight'])
print('\nWe see few outliers. Let us check the row with outliers value:\n')
plt.show()

sleep(sleep_time)

# Next we search for the outlier data by setting upper and lower bounds for the data
# Any point that does not lie between these values is considered to be useless
dfw = df['Weight']
dfw_Q1 = dfw.quantile(0.25)
dfw_Q3 = dfw.quantile(0.75)
dfw_IQR = dfw_Q3 - dfw_Q1
dfw_lowerend = dfw_Q1 - (1.5 * dfw_IQR)
dfw_upperend = dfw_Q3 + (1.5 * dfw_IQR)

dfw_outliers = dfw[(dfw < dfw_lowerend) | (dfw > dfw_upperend)]
print(dfw_outliers)
print("We see three rows with outliers value.",end='\n----------------------------------------\n')
sleep(sleep_time)


# 2. Length1	

# First plot the box plot
print("Plot a Box Plot for the length1 data")
sns.boxplot(x=df['Length1'])
plt.show()

sleep(sleep_time)

# Next we search for the outlier data by setting upper and lower bounds for the data
dflv = df['Length1']
dflv_Q1 = dflv.quantile(0.25)
dflv_Q3 = dflv.quantile(0.75)
dflv_IQR = dflv_Q3 - dflv_Q1
dflv_lowerend = dflv_Q1 - (1.5 * dflv_IQR)
dflv_upperend = dflv_Q3 + (1.5 * dflv_IQR)

dflv_outliers = dflv[(dflv < dflv_lowerend) | (dflv > dflv_upperend)]
print(dflv_outliers)
print("We see three rows with outliers value.",end='\n----------------------------------------\n')

sleep(sleep_time)

# 3. Length2	

# First plot the box plot
print("Plot a Box Plot for the length2 data")
sns.boxplot(x=df['Length2'])
plt.show()

# Next we search for the outlier data by setting upper and lower bounds for the data
dfdia = df['Length2']
dfdia_Q1 = dfdia.quantile(0.25)
dfdia_Q3 = dfdia.quantile(0.75)
dfdia_IQR = dfdia_Q3 - dfdia_Q1
dfdia_lowerend = dfdia_Q1 - (1.5 * dfdia_IQR)
dfdia_upperend = dfdia_Q3 + (1.5 * dfdia_IQR)

dfdia_outliers = dfdia[(dfdia < dfdia_lowerend) | (dfdia > dfdia_upperend)]
print(dfdia_outliers)
print("We see three rows with outliers value.",end='\n----------------------------------------\n')

sleep(sleep_time)

# 4. Length3	

# First plot the box plot
print("Plot a Box Plot for the length3 data")
sns.boxplot(x=df['Length3'])
plt.show()

# Next we search for the outlier data by setting upper and lower bounds for the data
dfcro = df['Length3']
dfcro_Q1 = dfcro.quantile(0.25)
dfcro_Q3 = dfcro.quantile(0.75)
dfcro_IQR = dfcro_Q3 - dfcro_Q1
dfcro_lowerend = dfcro_Q1 - (1.5 * dfcro_IQR)
dfcro_upperend = dfcro_Q3 + (1.5 * dfcro_IQR)

dfcro_outliers = dfcro[(dfcro < dfcro_lowerend) | (dfcro > dfcro_upperend)]
print(dfcro_outliers)
print("We see one row with an outlier value.",end='\n----------------------------------------\n')

sleep(sleep_time)

print("We see that all the outliers of the data set line in the row 142 to 144.==>\n",end='\n----------------------------------------\n')
print(df[142:145])

sleep(sleep_time)

print("So we drop those rows and use the remaining data",end='\n----------------------------------------\n')
# we use the drop() function to remove those data rows 
df1 = df.drop([142,143,144])

# We then look at the remaining data using the .describe() function
df1.describe().T

sleep(sleep_time)


# Next we go onto the data prediction part of the solution
# We first need to pick the dependent variable  ie the variable we need to predict 
# and then we need to choose the other independent variables

# Dependant (Target) Variable:
y = df1['Weight']
# Independant Variables:
X = df1.iloc[:,2:7]

# we use scikitlearns train test models to get the data needed to train a data model given the inputs of the 
# dependent and independent variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('X_train: ', np.shape(X_train))
print('y_train: ', np.shape(y_train))
print('X_test: ', np.shape(X_test))
print('y_test: ', np.shape(y_test))

print("\n----------------------------------------\n")



'''
	We will be doing multiple linear regression on the dataset 
		The Multiple Linear Regression Formula is :

		y = b0 + b1X1 + b2X2 + b3X3 + ... + bnXn

		where:
			y : Dependent variable
			b0 : Constant
			b1 - bn : Coefficients
			X1 - Xn : Independent variables
'''

# Next we will use Sklearn Library Linear Regression Model to train our dataset by using the 
# training data set we got previously

reg = LinearRegression()
reg.fit(X_train,y_train)

# My model's parameters:
print('Model intercept: ', reg.intercept_)
print('Model coefficients: ', reg.coef_)
print("\n----------------------------------------\n")


# We can now display the prediction formula for our linear regression below
print("The linear regression formula for the model we've trained is given by ==>\n")
print('y = ' + str('%.2f' % reg.intercept_) + ' + ' + str('%.2f' % reg.coef_[0]) + '*X1 ' + str('%.2f' % reg.coef_[1]) + '*X2 ' +
      str('%.2f' % reg.coef_[2]) + '*X3 + ' + str('%.2f' % reg.coef_[3]) + '*X4 + ' + str('%.2f' % reg.coef_[4]) + '*X5')
print("\n----------------------------------------\n")

# Next we use out regression to predict the data 
# The input will be the X_test array as our testing array
y_pred = reg.predict(X_test)

print(f"R-squared value for our prediction is :{r2_score(y_test, y_pred)}")


# After the prediction we can visualize the predictions

# 1: Linear Regression Model for Weight Estimation
plt.scatter(X_test['Length3'], y_test, color='red', alpha=0.4)
plt.scatter(X_test['Length3'], y_pred, color='blue', alpha=0.4)
plt.xlabel('Cross Length in cm')
plt.ylabel('Weight of the fish')
plt.title('Linear Regression Model for Weight Estimation')
plt.show()

# 2: Linear Regression Model for Weight Estimation

plt.scatter(X_test['Length1'], y_test, color='purple', alpha=0.5)
plt.scatter(X_test['Length1'], y_pred, color='orange', alpha=0.5)
plt.xlabel('Vertical Length in cm')
plt.ylabel('Weight of the fish')
plt.title('Linear Regression Model for Weight Estimation')
plt.show()

# 3: Linear Regression Model for Weight Estimation

plt.scatter(X_test['Length2'], y_test, color='purple', alpha=0.4)
plt.scatter(X_test['Length2'], y_pred, color='green', alpha=0.4)
plt.xlabel('Diagonal Length in cm')
plt.ylabel('Weight of the fish')
plt.title('Linear Regression Model for Weight Estimation')
plt.show()

# 4: Linear Regression Model for Weight Estimation

plt.scatter(X_test['Height'], y_test, color='orange', alpha=0.5)
plt.scatter(X_test['Height'], y_pred, color='blue', alpha=0.5)
plt.xlabel('Height in cm')
plt.ylabel('Weight of the fish')
plt.title('Linear Regression Model for Weight Estimation')
plt.show()

# 5: Linear Regression Model for Weight Estimation

plt.scatter(X_test['Width'], y_test, color='gray', alpha=0.5)
plt.scatter(X_test['Width'], y_pred, color='red', alpha=0.5)
plt.xlabel('Width in cm')
plt.ylabel('Weight of the fish')
plt.title('Linear Regression Model for Weight Estimation')
plt.show()

# Finally we just compare the real and the predicted weights

y_pred1 = pd.DataFrame(y_pred, columns=['Estimated Weight'])
y_test1 = pd.DataFrame(y_test)
y_test1 = y_test1.reset_index(drop=True)
ynew = pd.concat([y_test1, y_pred1], axis=1)
print(ynew)

print("From the results above, one can see there is a tendency towards errorous estimations when the weight is small.")