#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.datasets import churn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[39]:


data = pd.read_csv("churn.csv")
data.head()


# In[ ]:


import pandas as pd
xb, yb = data[return_x_y==True]
df_churn = pd.DataFrame(xb,columns==data().feature.names)
df_churn.head()


# In[ ]:





# In[15]:


data.info()


# In[20]:


data.describe()


# In[21]:


data.isna().sum()


# In[2]:


import pandas as pd
xb, yb = data[return xb_yb == True]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xb,yb, test_split=0.3, random_state=42)


# In[47]:





# In[4]:


import numpy as np
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


# In[5]:


x


# In[8]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 42)
model.fit(x, y)


# In[9]:


print("Classes: ", model.classes_)
print("Intercept: ",model.intercept_)
print("Coef: ",model.coef_)


# In[11]:


print("Probability: ",model.predict_proba(x))
model.predict(x)


# In[15]:


from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(y, model.predict(x))
import seaborn as sns

cm = confusion_matrix(y, model.predict(x))
sns.heatmap(cm, annot=True)


# In[14]:





# In[27]:


from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv("churn.csv")
xb, yb = data[return xb_yb == True]
x_train, x_test, y_train, y_test = train_test_split(xb,yb, test_split=0.3, random_state=42)
from sklearn.metrics import classification_report
print(classification_report(y_test, cv["estimator"][0].predict(X_test)))


# In[30]:


from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
models = LogisticRegression(random_state=42,n_jobs=-1)
cv = cross_validate(models,X_train,y_train,cv = 3, n_jobs=-1, return_estimator=True)

final_model = cv["estimator"][0]

y_pred = final_model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred))


# In[ ]:


# Using Decision tree


# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)

#read data
data = pd.read_csv("churn.csv")
data.head()


# In[4]:


data.info()


# In[3]:


data.describe()


# In[6]:


data.isna().sum()


# In[ ]:


#Data Preprocessing


# In[14]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

label_encoder = LabelEncoder()
data["Label"] = label_encoder.fit_transform(data["MonthlyCharge"]) 
data.head()


# In[8]:


data["Label"].value_counts()


# In[9]:


categories = list(label_encoder.inverse_transform([0, 1, 2]))
categories


# In[11]:


X, y = data.iloc[: , :-1], data.iloc[: , -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=4 , random_state=42)
clf.fit(X_train,y_train)
print("Accuracy of train:",clf.score(X_train,y_train))
print("Accuracy of test:",clf.score(X_test,y_test))


# In[ ]:


Bias for training = 1-0.21988 =0.78
Bias for testing = 1-0.19455 =0.81
variance = 0.03


# In[12]:


Visualization

import os
from sklearn.tree import export_graphviz
# We need to locate graphiz directory for visualization (after conda)
os.environ["PATH"] += ';' + r'C:\Users\Dell\Anaconda3\Library\bin\graphviz'

import graphviz

dot_data = export_graphviz(clf, out_file=None,
                     feature_names=X.columns,
                     class_names=categories,
                     filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph


# In[15]:


plt.figure(figsize=(12, 8))
importance = clf.feature_importances_
sns.barplot(x=importance, y=X.columns)
plt.show()


# In[16]:


importance


# In[17]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
pred = clf.predict(X_test)
print(classification_report(y_test,pred))


# In[18]:


print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("F1 Score = {}".format(f1_score(y_test, pred,average='macro')))


# In[ ]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm, square=True, annot=True, cbar=False)
ax.xaxis.set_ticklabels(categories, fontsize = 12)
ax.yaxis.set_ticklabels(categories, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




