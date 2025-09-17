#%%
import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import warnings
from graphviz import *
warnings.filterwarnings('ignore')
#%%
data=pd.read_csv('../data/StressLevelDataset.csv')
data_sd=pd.read_csv('StressLevelDataset.csv')
X=data.iloc[:,:20]
Y=data.iloc[:,20].values
ros=RandomOverSampler(random_state=42)
x_resampled,y_resampled= ros.fit_resample(X,Y)
rus = RandomUnderSampler(random_state=42)
x_rus_resampled, y_rus_resampled = rus.fit_resample(X, Y)


#%%
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
x_train_resam,x_test_res,y_train_resam,y_test_res=train_test_split(x_resampled,y_resampled,test_size=0.2,random_state=0)
x_train_rus,x_test_rus,y_train_rus,y_test_rus=train_test_split(x_rus_resampled,y_rus_resampled,test_size=0.2,random_state=0)
fig,(ax1,ax2,ax3) =plt.subplots(1,3,figsize=(12,8))
sns.countplot(x=y_train,ax=ax1)
ax1.set_title('Stress Level Dataset y_train before resampling')
plt.xticks(rotation=90)
ros=RandomOverSampler(random_state=42)
ax2.set_title('Stress Level Dataset y_train after resampling (RandomOverSampler)')
sns.countplot(x=y_train_resam,ax=ax2)
ax3.set_title('Stress Level Dataset y_train after resampling (RandomUnderSampler)')
sns.countplot(x=y_train_rus,ax=ax3)
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
#%%
xgb_train = xgb.DMatrix(x_train,y_train)
xgb_test = xgb.DMatrix(x_test,y_test)
#resampled
xgb_train_resam = xgb.DMatrix(x_train_resam,y_train_resam)
xgb_test_resam = xgb.DMatrix(x_test_res,y_test_res)
#unersampled
xgb_rus_train=xgb.DMatrix(x_train_rus,y_train_rus)
xgb_rus_test=xgb.DMatrix(x_test_rus,y_test_rus)
#%%
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'eta': 0.1,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1,
    'alpha': 0.1,
    'nthread': -1,
    'seed': 42
}
n=50
model=xgb.train(params=params,dtrain=xgb_train,num_boost_round=n)
model_resampled=xgb.train(params=params,dtrain=xgb_train_resam,num_boost_round=n)
model_rus=xgb.train(params=params,dtrain=xgb_rus_train,num_boost_round=n)
#%%
preds=model.predict(xgb_test)
preds=np.round(preds)
accuracy=accuracy_score(y_test,preds)
print('Accuracy before resampling:',accuracy*100,'%')
preds_res=model_resampled.predict(xgb_test_resam)
preds_res=np.round(preds_res)
accuracy_res=accuracy_score(y_test_res,preds_res)
print('Accuracy after resampling (Oversampling) :',accuracy_res*100,'%')
preds_rus=model_rus.predict(xgb_rus_test)
preds_rus=np.round(preds_rus)
accuracy_res=accuracy_score(y_test_rus,preds_rus)
print('Accuracy after resampling (Undesampling) :',accuracy_res*100,'%')
#%%
print(model.get_score(importance_type='gain'))
print('-'*50)
model.get_dump()
#%%
output_dir = '../stress level output/trees eg'
os.makedirs(output_dir, exist_ok=True)
for i in range(15):
    xgb.plot_tree(model,num_trees=i)
    print(f'saving tree {i+1}.png')
    plt.savefig(f'{output_dir}/tree{i+1}.png',dpi=300,bbox_inches='tight')
    plt.close()
#%%
output_dir='../stress level output/features importance'
os.makedirs(output_dir, exist_ok=True)
models=[(model,'base'),(model_resampled,'Oversampled'),(model_rus,'Undersampled')]
for model,name in models:
    xgb.plot_importance(model,title= f'Feature Importance in {name} model')
    plt.savefig(f'{output_dir}/importance {name}.png',dpi=300,bbox_inches='tight')
#%%
