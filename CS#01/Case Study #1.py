#!/usr/bin/env python
# coding: utf-8

# # Case Study #1

# ## 1. Case Study description

# Tasks:
# * Describe the dataset and any issues with it.
# * Generate a minimum of 5 unique visualizations using the data and write a brief description of your observations. Additionally, all attempts should be made to make the visualizations visually appealing
# * Create a feature set and create a model which predicts interest_rate using at least 2 algorithms. Describe any data cleansing that must be performed and analysis when examining the data.
# * Visualize the test results and propose enhancements to the model, what would you do if you had more time. Also describe assumptions you made and your approach.
# 

# ## 2. Dataset

# **Dataset:** Loan data from Lending Club 
# **Url:** https://www.openintro.org/data/index.php?data=loans_full_schema
# 
# **Description**  
# This data set represents thousands of loans made through the Lending Club platform, which is a platform that allows individuals to lend to other individuals. Of course, not all loans are created equal. Someone who is a essentially a sure bet to pay back a loan will have an easier time getting a loan with a low interest rate than someone who appears to be riskier. And for people who are very risky? They may not even get a loan offer, or they may not have accepted the loan offer due to a high interest rate. It is important to keep that last part in mind, since this data set only represents loans actually made, i.e. do not mistake this data for loan applications! 

#  **Variables:**
#  
#  | Variable | Description |  
#  | :- | :- |
#  | emp_title | Job title. |  
#  | emp_length | Number of years in the job, rounded down. | If longer than 10 years, then this is represented by the value 10. |
#  | state | Two-letter state code. |
#  | homeownership | The ownership status of the applicant's residence. |
#  | annual_income | Annual income. |
#  | verified_income | Type of verification of the applicant's income. |
#  | debt_to_income | Debt-to-income ratio. |
#  | annual_income_joint | If this is a joint application, then the annual income of the two parties applying. |
#  | verification_income_joint | Type of verification of the joint income. |
#  | debt_to_income_joint | Debt-to-income ratio for the two parties. |
#  | delinq_2y | Delinquencies on lines of credit in the last 2 years. |
#  | months_since_last_delinq | Months since the last delinquency. |
#  | earliest_credit_line | Year of the applicant's earliest line of credit
#  | inquiries_last_12m | Inquiries into the applicant's credit during the last 12 months. |
#  | total_credit_lines | Total number of credit lines in this applicant's credit history. |
#  | open_credit_lines | Number of currently open lines of credit. |
#  | total_credit_limit | Total available credit, e. |g. | if only credit cards, then the total of all the credit limits. | This excludes a mortgage. |
#  | total_credit_utilized | Total credit balance, excluding a mortgage. |
#  | num_collections_last_12m | Number of collections in the last 12 months. | This excludes medical collections. |
#  | num_historical_failed_to_pay | The number of derogatory public records, which roughly means the number of times the applicant failed to pay. |
#  | months_since_90d_late | Months since the last time the applicant was 90 days late on a payment. |
#  | current_accounts_delinq | Number of accounts where the applicant is currently delinquent. |
#  | total_collection_amount_ever | The total amount that the applicant has had against them in collections. |
#  | current_installment_accounts | Number of installment accounts, which are (roughly) accounts with a fixed payment amount and period. | A typical example might be a 36-month car loan. |
#  | accounts_opened_24m | Number of new lines of credit opened in the last 24 months. |
#  | months_since_last_credit_inquiry | Number of months since the last credit inquiry on this applicant. |
#  | num_satisfactory_accounts | Number of satisfactory accounts. |
#  | num_accounts_120d_past_due | Number of current accounts that are 120 days past due. |
#  | num_accounts_30d_past_due | Number of current accounts that are 30 days past due. |
#  | num_active_debit_accounts | Number of currently active bank cards. |
#  | total_debit_limit | Total of all bank card limits. |
#  | num_total_cc_accounts | Total number of credit card accounts in the applicant's history. |
#  | num_open_cc_accounts | Total number of currently open credit card accounts. |
#  | num_cc_carrying_balance | Number of credit cards that are carrying a balance. |
#  | num_mort_accounts | Number of mortgage accounts. |
#  | account_never_delinq_percent | Percent of all lines of credit where the applicant was never delinquent. |
#  | tax_liens | a numeric vector
#  | public_record_bankrupt | Number of bankruptcies listed in the public record for this applicant. |
#  | loan_purpose | The category for the purpose of the loan. |
#  | application_type | The type of application: either individual or joint. |
#  | loan_amount | The amount of the loan the applicant received. |
#  | term | The number of months of the loan the applicant received. |
#  | interest_rate | Interest rate of the loan the applicant received. |
#  | installment | Monthly payment for the loan the applicant received. |
#  | grade | Grade associated with the loan. |
#  | sub_grade | Detailed grade associated with the loan. |
#  | issue_month | Month the loan was issued. |
#  | loan_status | Status of the loan. |
#  | initial_listing_status | Initial listing status of the loan. | (I think this has to do with whether the lender provided the entire loan or if the loan is across multiple lenders. |)
#  | disbursement_method | Dispersement method of the loan. |
#  | balance | Current balance on the loan. |
#  | paid_total | Total that has been paid on the loan by the applicant. |
#  | paid_principal | The difference between the original loan amount and the current balance on the loan. |
#  | paid_interest | The amount of interest paid so far by the applicant. |
#  | paid_late_fees | Late fees paid by the applicant. |
# 

# ## 3. Analysis

# ### 3.1 Import libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB


# In[2]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ### 3.2 Data Exploration

# We will load the dataset into a pandas dataframe.

# In[3]:


filename = 'loans_full_schema.csv'
df = pd.read_csv(filename, sep=',')


# In[4]:


df.head(n=10)


# In[5]:


print(f"The dataset has {df.shape[0]} entries with {df.shape[1]} variables each.")


# In[6]:


print("The datase includes the following variables:")
print(df.dtypes)


# ### 3.2.1 Variables

# The dataset cobtains both numeric and categorical variables. 

# In[7]:


numeric_columns = df._get_numeric_data().columns
categorical_columns = list(set(df.columns) - set(numeric_columns))
print(f"The dataset contains {len(df.columns)} variables. In particular, it contains: \n- {len(numeric_columns)} numeric variables, and \n- {len(categorical_columns)} categorical variables.")


# #### Numeric Variables

# In[8]:


print(f"The data consists of {len(numeric_columns)} numeric variables.")
print(f"Numeric Variables: \n{', '.join(numeric_columns)}")


# The statistics for the numerical variables are shown in the following table.

# In[9]:


df[numeric_columns].describe()


# #### Categorical Variables

# In[10]:


print(f"The data consists of {len(categorical_columns)} categorical variables.")
print(f"Categorical Variables: \n{', '.join(categorical_columns)}")


# The statistics for the categorical variables are shown in the following table.

# In[11]:


df[categorical_columns].describe(include='all')


# ## 3.3 Data Preprocessing

# In this step, we will look in more depth into our dataset. We will check for:  
# - Missing NaN values
# - Duplicated rows  
# - Outliers  
# - Mismatch data

# ### 3.3.1 Missing Values

# In this step, we will deal with the NaN values of the dataset. 

# In[12]:


cols_with_nan = df.isnull().sum()
nan_columns = cols_with_nan.loc[cols_with_nan > 0].index.tolist()
cols_with_nan = cols_with_nan[nan_columns]
print("The columns of the dataset with NaN values are the following: ")
print(cols_with_nan)


# #### Handling Categorical Variables

# In[13]:


categorical_nan_columns = list(set(nan_columns).intersection(set(categorical_columns)))
print(f"The categorical variables with missing values are: \n{', '.join(categorical_nan_columns)}")


# In the following table, there are the summary of the categorical variables with missing values.
# * The "emp_title" variable is empty when the employer job's title is missing or individual is unemployed.
# * The "verification_income_joint" variable is empty when the application was submitted by one person (application type as individual) or the information is missing. 

# In[14]:


df[categorical_nan_columns].describe()


# In[15]:


df['verification_income_joint'].value_counts(dropna=False)


# In[16]:


df[['verification_income_joint','application_type']].value_counts(dropna=False)


# We will handle the two variables accordingly:
# * emp_title: We replace it with "Not Provided" if person has declared non-zero annual income. Otherwise, the value will be set as "Unemployed".
# * verification_income_joint: The NaN values will be replaced with "Not Verified" when the application is joint.

# In[17]:


df['verification_income_joint'] = np.where((df['verification_income_joint'].isnull()) & (df['application_type'] == 'joint'), "Not Verified", df['verification_income_joint'])
df['emp_title'] = np.where((df['emp_title'].isnull()) & (df['annual_income'] > 0), "Not Provided", df['emp_title'])
df['emp_title'] = np.where((df['emp_title'].isnull()) & (df['annual_income'] == 0), "Unemployed", df['emp_title'])


# In the following table, we can see that the combination NaN in verification_income_joint and joint in application_type does not exist anymore.

# In[18]:


df[['verification_income_joint','application_type']].value_counts(dropna=False)


# The summary for the categorical variables after processing are shown in the next table.

# In[19]:


df[categorical_nan_columns].describe()


# #### Handling Numerical Variables

# In[20]:


numeric_nan_columns = list(set(nan_columns).intersection(set(numeric_columns)))
print(f"Numerical variables with missing values: \n{', '.join(numeric_nan_columns)}")


# In[21]:


df[numeric_nan_columns].describe()


# #### Variable "emp_length"

# In[22]:


get_ipython().run_cell_magic('capture', '--no-stdout', "print(df[['emp_length','emp_title']][df['emp_length'].isnull()][df['emp_title'] == 'Not Provided'].value_counts(dropna=False))\nprint(df[['emp_length','emp_title']][df['emp_length'].isnull()][df['emp_title'] == 'Unemployed'].value_counts(dropna=False))")


# We will handle the emp_length variable in the following way based on the emp_title variable:
# * emp_title is "Not Provided": Do Nothing.
# * emp_title is "Unemployed": the emp_length value will be replaced with zero.

# In[23]:


df['emp_length'] = np.where((df['emp_length'].isnull()) & (df['emp_title'] == "Unemployed"), 0.0, df['emp_length'])
df['emp_length'] = np.where((df['emp_length'].isnull()) & (df['emp_title'] == "Not Provided"), df['emp_length'], df['emp_length'])


# In[24]:


get_ipython().run_cell_magic('capture', '--no-stdout', "print(df[['emp_length','emp_title']][df['emp_length'].isnull()][df['emp_title'] == 'Not Provided'].value_counts(dropna=False))\nprint(df[['emp_length','emp_title']][df['emp_length'].isnull()][df['emp_title'] == 'Unemployed'].value_counts(dropna=False))")


# #### Variables "annual_income_joint" & "debt_to_income_joint"

# In[25]:


get_ipython().run_cell_magic('capture', '--no-stdout', "print(df[['annual_income_joint','debt_to_income_joint', 'application_type']][df['annual_income_joint'].isnull()][df['debt_to_income_joint'].isnull()].value_counts(dropna=False))")


# The missing values for the above variables are referring only to the individual applications. There is no need for further examination.

# #### Variable "debt_to_income"

# In[26]:


get_ipython().run_cell_magic('capture', '--no-stdout', "print(df[['debt_to_income', 'annual_income']][df['debt_to_income'].isnull()].value_counts(dropna=False))\ndf['debt_to_income'] = np.where(df['debt_to_income'].isnull(), 0.0, df['debt_to_income'])")


# We replaced the missing values with zeros.

# #### Other Variables 
# * The columns "months_since_last_delinq" & "months_since_90d_late" & "months_since_last_credit_inquiry" will not be used, since a large percentage of their values are NaN.
# * The variable num_accounts_120d_past_due will not be used, since all non missing values are zeros.

# In[27]:


not_used_columns = ["months_since_last_delinq", "months_since_90d_late", "months_since_last_credit_inquiry", "num_accounts_120d_past_due"]


# In[28]:


df['num_accounts_120d_past_due'].sum()


# ### 3.3.2 Duplicated Rows

# In[29]:


print(f"The dataset contains {len(df[df.duplicated()])} duplicates rows.")


# ### 3.3.3 Handling Outliers

# We will ignore the variables 

# In[30]:


to_be_analysed = []
ignore_columns = ['emp_length', 'annual_income_joint', 'debt_to_income_joint', 'annual_income', 'debt_to_income']
max_threshold = 8000
min_threshold = 2000

df_filtered = df


# In[31]:


for col in numeric_columns:
    if col in not_used_columns or col in ignore_columns:
        continue
        
    q_low = df[col].quantile(0.01)
    q_hi  = df[col].quantile(0.99)

    df_filtered = df[(df[col] < q_hi) & (df[col] > q_low)]
    
    if df_filtered.shape[0] > max_threshold:
        continue
    elif df_filtered.shape[0] <= max_threshold:
        not_used_columns.append(col)
        continue
    else:
        to_be_analysed.append(col)
        print(f"{col}: next shape {df_filtered.shape}")
    
    


# In[32]:


print(f"The following Variables will not be used due to their distribution: \n\n{', '.join(not_used_columns)}")
print()
updated_columns = set(df.columns)-set(not_used_columns)
print(f"The updated Variables are: \n\n{', '.join(updated_columns)}")


# In[33]:


df[updated_columns].describe()


# #### 3.3.4 Reformat Variables

# We will create some new variables that will get the values from the existing columns depending if the application is joint or not: 
# * calc_annual_income   [annual_income_joint, annual_income]
# * calc_verified_income [verification_income_joint, verified_income]
# * calc_debt_to_income  [debt_to_income_joint, debt_to_income]

# In[34]:


df['calc_annual_income'] = np.where((df['application_type']=='joint'), df['annual_income_joint'], df['annual_income'])
df['calc_verified_income'] = np.where((df['application_type']=='joint'), df['verification_income_joint'], df['verified_income'])
df['calc_debt_to_income'] = np.where((df['application_type']=='joint'), df['debt_to_income_joint'], df['debt_to_income'])


# In[35]:


df[["calc_annual_income", "calc_debt_to_income"]].describe()


# In[36]:


df['calc_verified_income'].describe()


# In[37]:


numeric_columns = df._get_numeric_data().columns
categorical_columns = list(set(df.columns) - set(numeric_columns))
print(f"The dataset contains {len(df.columns)} variables. In particular, it contains: \n- {len(numeric_columns)} numeric variables, and \n- {len(categorical_columns)} categorical variables.")


# ### 3.4 Data Visualization

# In[38]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=11, ncols=4)
fig.set_size_inches(40, 19)
fig.tight_layout(pad=3.0)
for i in range(11):
    for j in range(4):
        index = numeric_columns[4*i+j]
        axes[i, j].title.set_text(index)
        axes[i, j].plot(df[index])


# In[39]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=11, ncols=4)
fig.set_size_inches(40, 19)
fig.tight_layout(pad=3.0)
for i in range(11):
    for j in range(4):
        index = numeric_columns[4*i+j]
        axes[i, j].title.set_text(index)
        axes[i, j].hist(df[index])


# In[40]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=11, ncols=4)
fig.set_size_inches(40, 19)
fig.tight_layout(pad=3.0)
for i in range(11):
    for j in range(4):
        index = numeric_columns[4*i+j]
        axes[i, j].title.set_text(index)
        axes[i, j].scatter(df[index], df['interest_rate'])


# In[41]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=11, ncols=4)
fig.set_size_inches(40, 19)
fig.tight_layout(pad=3.0)
for i in range(11):
    for j in range(4):
        index = numeric_columns[4*i+j]
        axes[i, j].title.set_text(index)
        axes[i, j].boxplot(df[index])


# ## 4. Prediction Problem

# ### 4.1 Output Variable: interest_rate

# In[42]:


unique_interest = df["interest_rate"].unique()
len(unique_interest)


# In[43]:


ax = df['interest_rate'].plot.hist(bins=24, alpha=0.5)


# In[44]:


df['sub_grade'].value_counts().plot(kind='bar')


# In[45]:


df['grade'].value_counts().plot(kind='bar')


# In[46]:


desired_columns = ["grade", "sub_grade", "interest_rate"]
_df = df[desired_columns]
_df = _df.drop_duplicates(ignore_index=True)
_df = _df.sort_values(by=desired_columns, ignore_index=True)
print(_df)


# ### 4.2 Split the dataset

# In[47]:


train = df.sample(frac=0.8, random_state=42) 
test = df.drop(train.index)
Y_train = train["interest_rate"]
Y_train_alt = train["sub_grade"]
Y_test_alt = test["sub_grade"]
Y_test = test["interest_rate"]
X_train = train.drop(['interest_rate'], axis=1)
X_test = test.drop(['interest_rate'], axis=1)
print(X_train.shape, X_test.shape)


# ### 4.3 Feature engineering

# In this step, we will:
# * Transform the categorical variables to numeric.
# * Scale the variables with Min-Max Normalization
# * Calculate the correlation between the variables and the "interest_rate"

# In[48]:


_numeric_columns = X_train._get_numeric_data().columns
_categorical_columns = list(set(X_train.columns) - set(_numeric_columns))

for col in _categorical_columns:
    if col == "emp_title":
        continue
    LE = LabelEncoder()
    X_train[col] = LE.fit_transform(X_train[col])
    X_test[col] = LE.transform(X_test[col])

X_train = X_train.drop(['emp_title'], axis=1)
X_test = X_test.drop(['emp_title'], axis=1)


# In[49]:


scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train.to_numpy())
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

Y_train_scaled = scaler.fit_transform(Y_train.to_numpy().reshape(-1, 1))
Y_train_scaled = series = pd.Series(np.squeeze(Y_train_scaled)) 

X_test_scaled = scaler.fit_transform(X_test.to_numpy())
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# In[50]:


corrM = X_train_scaled.corrwith(Y_train_scaled)
corrM.plot(kind='bar', figsize=(16, 7))


# In the graph above, we can see the correlation of the target variable with the variables of the test.  
# **We will not use the grade and sub_grade variables because this will be consider cheating. Because, the grade/subgrade is calculated from interest_rate.**

# In the following graphs, there are the variables with the higher positive/negative correlation with the target variable.

# In[51]:


corrM[corrM>0.12].plot(kind='bar')


# In[52]:


corrM[corrM<-0.12].plot(kind='bar')


# In[53]:


desired_cols = list(corrM[corrM>0.12].index) + list(corrM[corrM<-0.12].index)
desired_cols.remove("verified_income")
desired_cols.remove("annual_income_joint")
desired_cols.remove("debt_to_income")
desired_cols.remove("debt_to_income_joint")
desired_cols.remove("grade")
desired_cols.remove("sub_grade")
desired_cols.append("calc_annual_income")
desired_cols


# ### 4.4 Model Training & Evaluation

# In[54]:


X_train_scaled[desired_cols].isnull().sum().sum()
X_train_scaled = X_train_scaled[desired_cols]
X_test_scaled = X_test_scaled[desired_cols]


# ### 4.5.1 Baseline

# Our baseline is the result of always guessing the mean of the train target variable. Any successful model must be better than guessing the mean value.

# In[55]:


print("The baseline MSE error is {:.4f}.".format(mean_squared_error(Y_test, [Y_train.mean()] * len(Y_test))))


# ### 4.5.2 SVR Algorithm

# In[56]:


regr = svm.SVR()
regr.fit(X_train_scaled, Y_train)
y_predicted = regr.predict(X_test_scaled)
mean_squared_error(Y_test, y_predicted)


# This is the best attempt with 14.3880 in comparison with 23.4864.

# ### 4.5.3 BayesianRidge

# In[57]:


regr = BayesianRidge()
regr.fit(X_train_scaled, Y_train)
y_predicted = regr.predict(X_test_scaled)
mean_squared_error(Y_test, y_predicted)


# ### 4.5.4 LinearRegression

# In[58]:


regr = LinearRegression()
regr.fit(X_train_scaled, Y_train)
y_predicted = regr.predict(X_test_scaled)
mean_squared_error(Y_test, y_predicted)


# ### 4.5.5 As Classification problem

# In[59]:


clf = svm.SVC(gamma=10, kernel='rbf', C=1, coef0=0.0)
clf.fit(X_train_scaled, Y_train_alt)
y_predicted = clf.predict(X_test_scaled)


# In[60]:


_train = df[['interest_rate','sub_grade']].value_counts(dropna=False)
for val_index in range(len(y_predicted)):
    _val = _train[:, y_predicted[val_index]].index.values.mean()
    y_predicted[val_index] = _val
    
mean_squared_error(Y_test, y_predicted)


# In[61]:


mean_squared_error(Y_test, y_predicted)


# ### 4.5.6 Bayes

# In[62]:


gnb = GaussianNB()
y_predicted = gnb.fit(X_train_scaled, Y_train_alt).predict(X_test_scaled)


# In[63]:


_train = df[['interest_rate','sub_grade']].value_counts(dropna=False)
for val_index in range(len(y_predicted)):
    _val = _train[:, y_predicted[val_index]].index.values.mean()
    y_predicted[val_index] = _val
    
mean_squared_error(Y_test, y_predicted)


# ## 5. Comments

# There are some points worth mentioning:
# * The dataset contains only successful deals and not unsuccessful deals. So, you can not use it to predict if someone will get a loan or not.
# * The dataset includes some information after the deal is made. We can not use the extra information to predict the interest rate if we want to predict the interest rate in a successful deal. In this case, the cases that the applicant refused the loan because the higher interest rate is not included.
# * If the application was joint application, then the joint income, debt, etc was used for the calculations.
# * Many variables have a lot of outlier, or NaN values.
# * There is low correlation between interest rate and the other variables.
# 

# Possible Directions:  
# 1) Regression problem  
# 2) Classification problem with subset - check if all subset have same value for interest  

# ## 6. Improvements

# For future development:
# * More models/algorithms
# * Parameter Models Optimization.
# * Feature Engineering ALgorithms like PCA.

# In[ ]:




