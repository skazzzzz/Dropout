# %%
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

# %%
# Load the dataset
data = pd.read_csv('dataset.csv')
#pd.set_option('display.max_columns', None)
data.head()

# %%
data.shape

# %%
data.columns

# %%
# Get a summary of the dataset
data.info()

# %%
# Get basic statistics of numerical features
data.describe()

# %%
# Check for missing values
data.isnull().sum()

# %%
data = data.rename(columns={'Nacionality': 'Nationality'})
data.info()

# %%
data['Target'].value_counts()

# %%
#store the value counts in a variable
data_target = data['Target'].value_counts()

#visualize the target variable
plt.pie(data_target, labels=data_target.index, autopct='%2.1f%%')
plt.title('Target Distribution of Dataset in %')
plt.show()

# %%
data['Gender'].value_counts()

# %%
#15 gender affecting academic status of students
sns.countplot(data=data, x='Gender', hue='Target', hue_order=['Enrolled', 'Graduate', 'Dropout'])

plt.xticks(ticks=[0,1], labels=['Female', 'Male'])
plt.ylabel('Number of Students')
plt.show()

# %%
#0 Marital Status affecting the academic success of students
pd.crosstab(data["Marital status"], data["Target"], normalize='index').plot(kind="bar", figsize=(10,4), title="Marrital Status and Academic success", )
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=['Single', 'Married', 'Widower', 'Divorced', 'Facto Union', 'Legally Seperated'])

plt.xlabel('Marital Status')
plt.ylabel('Number of Students')
plt.legend(loc=9)
plt.show()

# %% [markdown]
# from the above bar graph:
# - Students who are legally seperated stand a high chance of dropping out.
# - Single students have more chance of graduating as their focus
# - Marriend and divorced individuals have more chances of dropping out but the gap between dropping out and graduating is small.

# %%
#11 Displaced students and the academic success of students
sns.countplot(data=data, x='Displaced', hue='Target', hue_order=['Enrolled', 'Dropout', 'Graduate'])

plt.xticks(ticks=[0,1], labels=['No','Yes'])
plt.ylabel('Number of Students')
plt.show()

# %% [markdown]
# 

# %% [markdown]
# This shows that students enroll between the ages 17 and 70 with the mean age at age 23

# %%
#12 Educational special Needs

sns.countplot(data=data, x='Educational special needs', hue='Target', hue_order=['Enrolled', 'Dropout', 'Graduate'])

plt.xticks(ticks=[0,1], labels=['No','Yes'])
plt.xlabel('Educational Special Needs')
plt.ylabel('Number of Students')
plt.show()

# %%
#13 Debtor
sns.countplot(data=data, x="Debtor", hue='Target', hue_order=['Dropout', 'Enrolled', 'Graduate'])

plt.xticks(ticks=[0,1], labels=['No','Yes'])
plt.xlabel('Debtor')
plt.ylabel('Number of Students')
plt.show()

# %%
#14 Tuition fees up to date

sns.countplot(data=data, x="Tuition fees up to date", hue='Target', hue_order=['Dropout', 'Enrolled', 'Graduate'])

plt.xticks(ticks=[0,1], labels=['No','Yes'])
plt.xlabel('Tuition Fees Up to Date')
plt.ylabel('Number of Students')
plt.show()

# %%
  #17 Age 
sns.displot(data=data, x='Age at enrollment', kde=True)
data['Age at enrollment'].describe()

plt.xlabel('Age at Enrolment')
plt.ylabel('Number of Students')
plt.show()

# %%
# Scholarship holder

sns.countplot(data=data, x="Scholarship holder", hue='Target', hue_order=['Dropout', 'Enrolled', 'Graduate'])

plt.xticks(ticks=[0,1], labels=['No','Yes'])
plt.xlabel('Scholarship Holder')
plt.ylabel('Number of Students')
plt.show()


# %%



