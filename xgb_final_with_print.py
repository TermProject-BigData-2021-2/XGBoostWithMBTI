import pandas as pd
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import plotly.express as px
import warnings
import numpy
warnings.filterwarnings('ignore')
data=pd.read_csv('test_html_alt.csv')
print(data.head())


my_x = data.drop(['aaa'], axis=1)
my_y = data['aaa']

train_post,test_post, train_target, test_target = train_test_split(my_x,my_y, test_size = 0.2)

#print(train_data.head())
models_accuracy={}
n_estimators = 10
n_jobs = -1

param_test1 = {
         'max_depth':range(3,10,2),
          'min_child_weight':range(1,6,2)
          }
#M_depth = [3,5,7,9]
#M_child = [1,3,5]

subsample = [i/10.0 for i in range(5,10)]
colsample_bytree = [i  for i in range(1,3)]

score = 0
best_score = 0
best_parameter = {}
#for i in subsample:
#for j in colsample_bytree:
print("start new data")
"""
for i in subsample:
#for j in colsample_bytree:
    model = XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_depth=5,n_estimators=50,learning_rate=0.1,n_jobs=3,min_child_weight=1,gamma= 0, subsample = i, colsample_bytree = 1)
    scores = cross_val_score(model,my_x,my_y,cv=3, n_jobs = 3)
    score = numpy.mean(scores)
    if score > best_score:
        best_score = score
        best_parameter = {'subsample': i, 'max_depth':5, 'min_child':1, 'gamma':0, 'colsample_bytree':1}
        f = open("last_last_xgb_parameter.txt", 'a')
        f.write("\n" + str(best_score))
        f.write("\n" + str(best_parameter))
        f.close()
    
"""

#model = XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_depth=6,n_estimators=50,learning_rate=0.1,min_child_weight=1, gamma=0, subsample = 0.3, colsample_bytree = 1)
model = XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_depth=3,gamma=1.2,n_estimators=50,learning_rate=0.1,min_child_weight=3,subsample=0.6, n_jobs=1)

#model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators,  max_samples=1./n_estimators , n_jobs=n_jobs)
model.fit(train_post,train_target)
#f = open('xgb test_result.txt','a')
#f.write('\n"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""')
#f.write("\n"+str(best_score))
#f.write("\n"+str(best_parameter))
train_predict = model.predict(train_post)
test_predict = model.predict(test_post)
print('\n\nprediction')
print(test_predict[0:10])
print('\n\nreal')
print(test_target[0:10])
print('trainclassification report \n ',classification_report(train_target,train_predict))
print('test classification report \n',classification_report(test_target,test_predict))
models_accuracy['xgb']=accuracy_score(test_target,model.predict(test_post))
"""
f = open('xgb_test_result.txt', 'a')
f.write("\nxgb train report"+"\n")
f.write(classification_report(train_target,train_predict))
f.write("test report" +" \n")
f.write(classification_report(test_target,test_predict))
f.close()
"""
