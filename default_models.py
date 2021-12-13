
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
data=pd.read_csv('test.csv')
print(data.head())


my_x = data.drop(['aaa'], axis=1)
my_y = data['aaa']

train_post,test_post, train_target, test_target = train_test_split(my_x,my_y, test_size = 0.2)

#print(train_data.head())
models_accuracy={}
n_estimators = 10
n_jobs = -1


estimator = LinearSVC()
model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators,  max_samples=1./n_estimators , n_jobs=n_jobs)
model.fit(train_post,train_target)
train_predict = model.predict(train_post)
test_predict = model.predict(test_post)
print('trainclassification report \n ',classification_report(train_target,train_predict))
print('test classification report \n',classification_report(test_target,test_predict))
models_accuracy['linear svm']=accuracy_score(test_target,model.predict(test_post))
f = open('test_result_not_alt.txt', 'w')
f.write("\nlinear svm train report"+"\n")
f.write(classification_report(train_target,train_predict))
f.write("test report" +" \n")
f.write(classification_report(test_target,test_predict))
f.write(str(numpy.mean(cross_val_score(model,my_x,my_y,cv=3, n_jobs=-1))))
f.close()

"""
parameters = {
            'alpha': [ 0.001, 0.005, 0.01, 0.05,0.1,0.2,0.4, 0.25, 0.5 ,1,2,3,4,10,15,20,25],
                'fit_prior': [True, False]
                }
estimator = MultinomialNB(alpha = 0.1, fit_prior = True)
model = GridSearchCV(estimator, parameters, n_jobs=3, cv=2)
#model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators,  max_samples=1./n_estimators , n_jobs=n_jobs)
model.fit(train_post,train_target)
#train_predict = model.predict(train_post)
#test_predict = model.predict(test_post)
#print('trainclassification report \n ',classification_report(train_target,train_predict))
#print('test classification report \n',classification_report(test_target,test_predict))
#models_accuracy['NB']=accuracy_score(test_target,model.predict(test_post))
f = open('test_result_nb_tuning.txt', 'w')
f.write("\n" + str(model.best_params_))
f.write("\n" + str(model.best_score_))
#f.write("\nnb train report"+"\n")
#f.write(classification_report(train_target,train_predict))
#f.write("test report" +" \n")
#f.write(classification_report(test_target,test_predict))
f.close()
print(str(model.best_params_))
print(str(model.best_score_))
"""

model = LogisticRegression(max_iter=3000,C=0.5,n_jobs=-1)
#model= BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators,  max_samples=1./n_estimators , n_jobs=n_jobs)
model.fit(train_post,train_target)
train_predict = model.predict(train_post)
test_predict = model.predict(test_post)
print('trainclassification report \n ',classification_report(train_target,train_predict))
print('test classification report \n',classification_report(test_target,test_predict))
models_accuracy['log']=accuracy_score(test_target,model.predict(test_post))
f = open('test_result_not_alt.txt', 'a')
f.write("\nlog train report"+"\n")
f.write(classification_report(train_target,train_predict))
f.write("test report" +" \n")
f.write(classification_report(test_target,test_predict))
f.write(str(numpy.mean(cross_val_score(model,my_x,my_y,cv=3, n_jobs=-1))))

f.close()

model = DecisionTreeClassifier(max_depth=23, min_samples_leaf=8)
#model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators,  max_samples=1./n_estimators , n_jobs=n_jobs)
model.fit(train_post,train_target)
train_predict = model.predict(train_post)
test_predict = model.predict(test_post)
print('trainclassification report \n ',classification_report(train_target,train_predict))
print('test classification report \n',classification_report(test_target,test_predict))
models_accuracy['tree']=accuracy_score(test_target,model.predict(test_post))
f = open('test_result_not_alt.txt', 'a')
f.write("\ntree train report"+"\n")
f.write(classification_report(train_target,train_predict))
f.write("test report" +" \n")
f.write(classification_report(test_target,test_predict))
f.write(str(numpy.mean(cross_val_score(model,my_x,my_y,cv=3, n_jobs=-1))))
f.close()

model = RandomForestClassifier(max_depth=10)
#model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators,  max_samples=1./n_estimators , n_jobs=n_jobs)
model.fit(train_post,train_target)
train_predict = model.predict(train_post)
test_predict = model.predict(test_post)
print('trainclassification report \n ',classification_report(train_target,train_predict))
print('test classification report \n',classification_report(test_target,test_predict))
models_accuracy['forest']=accuracy_score(test_target,model.predict(test_post))
f = open('test_result_not_alt.txt', 'a')
f.write("\nforest train report"+"\n")
f.write(classification_report(train_target,train_predict))
f.write("test report" +" \n")
f.write(classification_report(test_target,test_predict))
f.write(str(numpy.mean(cross_val_score(model,my_x,my_y,cv=3, n_jobs=-1))))

f.close()


model = XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_depth=5,n_estimators=50,learning_rate=0.1, n_jobs=1)
#model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators,  max_samples=1./n_estimators , n_jobs=n_jobs)
model.fit(train_post,train_target)
train_predict = model.predict(train_post)
test_predict = model.predict(test_post)
print('trainclassification report \n ',classification_report(train_target,train_predict))
print('test classification report \n',classification_report(test_target,test_predict))
models_accuracy['xgb']=accuracy_score(test_target,model.predict(test_post))
f = open('test_result_not_alt.txt', 'a')
f.write("\nxgb train report"+"\n")
f.write(classification_report(train_target,train_predict))
f.write("test report" +" \n")
f.write(classification_report(test_target,test_predict))
f.write(str(numpy.mean(cross_val_score(model,my_x,my_y,cv=3, n_jobs=-1))))
f.close()
"""
model = CatBoostClassifier(loss_function='MultiClass',eval_metric='MultiClass',task_type='GPU',verbose=False)
#model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators,  max_samples=1./n_estimators , n_jobs=n_jobs)
model.fit(train_post,train_target)
train_predict = model.predict(train_post)
test_predict = model.predict(test_post)
print('trainclassification report \n ',classification_report(train_target,train_predict))
print('test classification report \n',classification_report(test_target,test_predict))
models_accuracy['cat']=accuracy_score(test_target,model.predict(test_post))
f = open('test_result_alt.txt', 'a')
f.write("\ncat train report"+"\n")
f.write(classification_report(train_target,train_predict))
f.write("test report" +" \n")
f.write(classification_report(test_target,test_predict))
f.close()
"""
f = open('test_result_not_alt.txt', 'a')
f.write("\n")
f.write(str(models_accuracy))
f.close()
print(models_accuracy)
