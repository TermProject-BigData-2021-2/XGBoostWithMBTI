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
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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
data=pd.read_csv('test.csv')
print(data.head())

my_x = data.drop(['aaa'], axis=1)
my_y = data['aaa']

train_post,test_post, train_target, test_target = train_test_split(my_x,my_y, test_size = 0.2)
"""
param_grid={'gpu_id' : [0],'tree_method':['gpu_hist'], 'booster' :['gbtree'],
                 'silent':[True],
                 'max_depth':[8],
                 'learning_rate':[0.01,0.1,0.5,1]
                 'gamma':[0,1,2,3],
                 'nthread':[3],
                 'colsample_bytree':[0.5,0.8],
                 'colsample_bylevel':[0.9],
                 'n_estimators':[100],
                 'objective':['binary:logistic'],
'random_state':[2]}
"""
#gcv= XGBClassifier(n_estimator=400, learning_rate = 0.1, max_depth=5, gamma=0, subsample = 0.8)

gcv = xgb1 = XGBClassifier(
            learning_rate =0.1,
                n_estimators=1000,
                    max_depth=5,
                        min_child_weight=1,
                            gamma=0,
                                subsample=0.8,
                                    colsample_bytree=0.8,
                                        objective= 'binary:logistic',
                                            nthread=-1,
                                                scale_pos_weight=1,
                                                    seed=2019
                                                    )
"""scorings = ['accuracy', 'f1_macro']
#cv=KFold(n_splits=6, random_state=1)
gcv=GridSearchCV(model, param_grid=param_grid, cv=2, n_jobs=3, verbose=2)
"""

gcv.fit(train_post,train_target)
train_predict = model.predict(train_post)
test_predict = model.predict(test_post)
print('trainclassification report \n ',classification_report(train_target,train_predict))
print('test classification report \n',classification_report(test_target,test_predict))

"""f = open("parameter.txt", "w")
f.write(str(gcv.best_params_))
f.write("\n")
f.write(str(gcv.best_score_))
        
print('final params', gcv.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv.best_score_)      # 최고의 점수
"""



