# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:08:54 2019

@author: RAhul
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import xgbclassifier
#from xgboost import XGBClassifier
#import xgboost as xgb
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

df = pd.read_csv('cybersecurity_training.csv', sep='|', index_col=False)

#print(df.info())

#df.to_excel('C:\\Users\\lenova\\Desktop\\coursera material\\Rahul asst\\knowledge_pit\\cybersecurity_training\\sample.xlsx')
list(df.columns)
missing_data = df.isnull()

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")  
#----------------------------------------------------------------------------------------------------
    
feature = df[[
 'categoryname',
 'ipcategory_name',
 'ipcategory_scope',
 'parent_category',
 'grandparent_category',
 'overallseverity',
 'weekday',
 'correlatedcount',
 'n1',
 'n2',
 'n3',
 'n4',
 'n5',
 'n6',
 'n7',
 'n8',
 'n9',
 'n10',
 'score',
 'srcip_cd',
 'dstip_cd',
 'srcport_cd',
 'dstport_cd',
 'alerttype_cd',
 'direction_cd',
 'eventname_cd',
 'severity_cd',
 'reportingdevice_cd',
 'devicetype_cd',
 'devicevendor_cd',
 'domain_cd',
 'protocol_cd',
 'username_cd',
 'srcipcategory_cd',
 'dstipcategory_cd',
 'isiptrusted',
 'untrustscore',
 'flowscore',
 'trustscore',
 'enforcementscore',
 'dstipcategory_dominate',
 'srcipcategory_dominate',
 'dstportcategory_dominate',
 'srcportcategory_dominate',
 'thrcnt_month',
 'thrcnt_week',
 'thrcnt_day',
 'p6',
 'p9',
 'p5m',
 'p5w',
 'p5d',
 'p8m',
 'p8w',
 'p8d']]

#----------------------------------------Filling missing values--------------------------------------------------------

#mode_feature = feature[[ 'n1',
# 'n2',
# 'n3',
# 'n4',
# 'n5',
# 'n6',
# 'n7',
# 'n8',
# 'n9',
# 'n10',
# 'score']].astype("float").mode(axis=0)

feature.replace("", np.nan, inplace = True)

feature['score'].value_counts()

feature['score'].fillna(1, inplace = True)

feature.fillna(0, inplace = True)

#feature[[ 'n1',
# 'n2',
# 'n3',
# 'n4',
# 'n5',
# 'n6',
# 'n7',
# 'n8',
# 'n9',
# 'n10',
# 'score']].replace(np.nan, mode_feature, inplace=True)


#-----------------Onehot Encoding-----------------------------------------------------------------------------

feature = pd.concat([feature,pd.get_dummies(df[['categoryname','ipcategory_name','ipcategory_scope',
                                                    'grandparent_category','weekday','dstipcategory_dominate',
                                                    'srcipcategory_dominate']])], axis=1)


feature.drop(['categoryname','ipcategory_name','ipcategory_scope',
                                                    'grandparent_category','weekday','dstipcategory_dominate',
                                                    'srcipcategory_dominate'],axis=1, inplace=True)

#--------------------------Data Preprocessing------------------------------------------------------------
X = feature
y = df['notified']


#__________________________________________________________________________________________________

#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#X_test.fillna(X_train.mode(), inplace=True)

# Feature Scaling---------------------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#----------------------------------------Decision Tree-----------------------------------------------
DT_model = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
DT_model.fit(X_train,y_train)

DT_yhat = DT_model.predict(X_test)
DT_yhat_all = DT_model.predict(X)
print("DT Jaccard index: %.2f" % jaccard_similarity_score(y_test, DT_yhat))
print("DT F1-score: %.2f" % f1_score(y_test, DT_yhat, average='weighted') )

#--------------------------
pipeline = Pipeline([('transformer', sc), ('estimator', DT_model)])

cv = KFold(n_splits=4)
scores = cross_val_score(pipeline, X, y, cv = cv)
print(scores)

#----------------------------------SVM-----------------------------------------------------

SVM_model = svm.SVC()
SVM_model.fit(X_train, y_train) 

SVM_yhat = SVM_model.predict(X_test)
print("SVM Jaccard index: %.2f" % jaccard_similarity_score(y_test, SVM_yhat))
print("SVM F1-score: %.2f" % f1_score(y_test, SVM_yhat, average='weighted'))

#-----------------------------------------------Logistic Regression----------------------------
LR_model = LogisticRegression(C=0.01).fit(X_train,y_train)
LR_yhat = LR_model.predict(X_test)
LR_yhat_prob = LR_model.predict_proba(X_test)
print("LR Jaccard index: %.2f" % jaccard_similarity_score(y_test, LR_yhat))
print("LR F1-score: %.2f" % f1_score(y_test, LR_yhat, average='weighted') )
print("LR LogLoss: %.2f" % log_loss(y_test, LR_yhat_prob))

# ---------------------------------------------KNN algorithm--------------------------------------------
# Best k
Ks=15
mean_acc=np.zeros((Ks-1))
std_acc=np.zeros((Ks-1))
ConfustionMx=[];
for n in range(1,Ks):
    
    #Train Model and Predict  
    kNN_model = KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat = kNN_model.predict(X_test)
    
    
    mean_acc[n-1]=np.mean(yhat==y_test);
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc

k = 5
#Train Model and Predict  
kNN_model = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

knn_yhat = kNN_model.predict(X_test)
print("KNN Jaccard index: %.2f" % jaccard_similarity_score(y_test, knn_yhat))
print("KNN F1-score: %.2f" % f1_score(y_test, knn_yhat, average='weighted'))






#-----------------------------XGboost----------------------------------------
#xgb_model = xgb.XGBClassifier() 
#xgb_model.fit(X_train, y_train) 
   
# Predicting the Test set results 
#xgb_yhat = xgb_model.predict(X_test) 
#print("XGB Jaccard index: %.2f" % jaccard_similarity_score(y_test, xgb_yhat))
#print("XGB F1-score: %.2f" % f1_score(y_test, xgb_yhat, average='weighted') )
# Making the Confusion Matrix 
#from sklearn.metrics import confusion_matrix 
#cm = confusion_matrix(y_test, y_pred) 

#----------------------------------Model Ealuation-------------------------------------------------------

# initialize list of lists 
data = [['Decision Tree', jaccard_similarity_score(y_test, DT_yhat), f1_score(y_test, DT_yhat, average='weighted'),0],
         ['SVM', jaccard_similarity_score(y_test, SVM_yhat), f1_score(y_test, SVM_yhat, average='weighted'),0],
         ['Logistic Regression', jaccard_similarity_score(y_test, LR_yhat), f1_score(y_test, LR_yhat, average='weighted'),log_loss(y_test, LR_yhat_prob)],
         ['KNN',jaccard_similarity_score(y_test, knn_yhat), f1_score(y_test, knn_yhat, average='weighted'),0]]
         #['XGB',jaccard_similarity_score(y_test, xgb_yhat),f1_score(y_test, xgb_yhat, average='weighted') ]] 
  
# Create the pandas DataFrame 
scores = pd.DataFrame(data, columns = ['Algorithm','Jaccard-score', 'F1-score','LogLoss']) 
  
# print dataframe. 
print(scores)




