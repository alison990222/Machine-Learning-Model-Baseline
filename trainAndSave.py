## python3 questionnair_train.py --questionnair ml.csv --label label
import pandas as pd
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
import pickle

def parse_arg():
	parser = argparse.ArgumentParser()
	parser.add_argument("--questionnair", required=True, dest = 'filename', help="questionnair csv")
	parser.add_argument("--label",required=True, dest = 'label', help="label column")
	args = parser.parse_args()
	return args


def readData():
	filename = args.filename
	data = pd.read_csv(filename,index_col=0)

	## split the data into test and train 

	X = data.drop(args.label,axis=1) ## 'label' for whitening
	y = data[args.label]
	# split the data into val and train
	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.30, random_state=10)
	return X,y,X_train, X_test, y_train, y_test 


def xgboostModel(X, y, X_train, X_test, y_train, y_test ):
	from xgboost import XGBRegressor,XGBClassifier
	from xgboost import plot_importance
	print("xgboost model processing")
	# Create the parameter grid: gbm_param_grid 
	gbm_param_grid = {
	    'n_estimators': range(1,15), # 樹有幾棵
	    'max_depth': range(1,10), # 樹的深度
	    'learning_rate': [0.01,0.03,0.1, 0.3,0.4, 0.45, 0.5, 0.55, 0.6],
	    'colsample_bytree': [.6, .7, .8, .9, 1],
	    'min_child_weight':range(1,6,2)
	    }

	gbm = XGBClassifier() 

	xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid,
	                   estimator=gbm,
	                   scoring="accuracy",verbose=False,
	                   n_iter=40, 
	                   cv=4) # cross validation

	xgb_random.fit(X, y)

	# # Print the best parameters
	# print("best estimator: ", xgb_random.best_estimator_)
	# print("Best parameters found: ", xgb_random.best_params_)
	# print("Best accuracy found: ", xgb_random.best_score_)

	xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
	       colsample_bytree=xgb_random.best_params_['colsample_bytree'], gamma=0, 
	       learning_rate=xgb_random.best_params_['learning_rate'], max_delta_step=0,
	       max_depth=9, min_child_weight=xgb_random.best_params_['min_child_weight'], missing=None, 
	       n_estimators=xgb_random.best_params_['n_estimators'],
	       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
	       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
	       silent=True, subsample=1)
	xgb.fit(X_train, y_train)
	xgbScore = round(xgb.score(X_test, y_test),3)
	# print(xgbScore)

	# # Print classification report for y_test
	xgb_pred = xgb.predict(X_test)
	# save model
	with open('model/xgboost.pickle', 'wb') as f:
		pickle.dump(xgb, f)

	return xgb_pred, xgbScore


def knnModel(X, y, X_train, X_test, y_train, y_test ):
	from sklearn.neighbors import KNeighborsClassifier
	print("knn model processing")
	## trying out multiple values for k
	k_range = range(1,50)
	weights_options=['uniform','distance']
	param = {'n_neighbors':k_range, 'weights':weights_options}
	grid = GridSearchCV(KNeighborsClassifier(), param,cv=4,verbose = False)
	## Fitting the model. 
	grid.fit(X,y)

	# print(grid.best_score_)
	# print(grid.best_params_)
	# print(grid.best_estimator_)

	knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
	                     metric_params=None, n_jobs=None, n_neighbors=grid.best_params_['n_neighbors'], p=2,
	                     weights=grid.best_params_['weights'])

	knn.fit(X_train, y_train)
	knnScore = round(knn.score(X_test, y_test),3)
	# print(knnScore)

	# # Print classification report for y_test
	knn_pred = knn.predict(X_test)
	
	# save model
	with open('model/knn.pickle', 'wb') as f:
		pickle.dump(knn, f)
	return knn_pred, knnScore

def gaussianModel(X, y, X_train, X_test, y_train, y_test ):
	# Gaussian Naive Bayes
	from sklearn.naive_bayes import GaussianNB,BernoulliNB, MultinomialNB
	from sklearn.metrics import accuracy_score
	print("gaussian model processing")
	gaussian = BernoulliNB()#MultinomialNB()#GaussianNB() 
	gaussian.fit(X, y)
	y_pred = gaussian.predict(X_test)
	gaussian_accy = round(accuracy_score(y_test, y_pred), 3)
	# print(gaussian_accy)

	# # Print classification report for y_test
	gaussian_pred = gaussian.predict(X_test)
	# save model
	with open('model/gaussian.pickle', 'wb') as f:
		pickle.dump(gaussian, f)
	return gaussian_pred, gaussian_accy

def logisticRegressionModel(X, y, X_train, X_test, y_train, y_test ):
	
	from sklearn.linear_model import LogisticRegression
	print("LogisticRegression model processing")
	C_vals = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]
	## Choosing penalties(Lasso(l1) or Ridge(l2))
	penalties = ['l1','l2']

	param = {'penalty': penalties, 'C': C_vals}

	logreg = LogisticRegression(solver='liblinear')
	## Calling on GridSearchCV object. 
	grid = GridSearchCV(estimator=LogisticRegression(), 
	                           param_grid = param,
	                           scoring = 'accuracy',
	                           cv = 4)
	## Fitting the model
	grid.fit(X, y)

	## Getting the best of everything. 
	# print (grid.best_score_)
	# print (grid.best_params_)
	# print(grid.best_estimator_)
	lr = LogisticRegression(penalty=grid.best_params_['penalty'], dual=False,  
	                          tol=0.0001, C=grid.best_params_['C'], fit_intercept=True, intercept_scaling=1, 
	                             class_weight=None, random_state=None, solver='liblinear', 
	                             max_iter=300, multi_class='ovr', verbose=False, warm_start=False)

	lr.fit(X_train, y_train)
	lrScore = round(lr.score(X_test, y_test),3)
	# print(lrScore)

	# # Print classification report for y_test
	lr_pred = lr.predict(X_test)
	# save model
	with open('model/LogisticRegression.pickle', 'wb') as f:
		pickle.dump(lr, f)

	return lr_pred, lrScore 

def randomForestModel(X, y, X_train, X_test, y_train, y_test ):
	from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
	from sklearn.ensemble import RandomForestClassifier
	print("random forest model processing")
	n_estimators = [140,145,150,155,160]
	max_depth = range(1,10)
	criterions = ['gini', 'entropy']
	cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)


	parameters = {'n_estimators':n_estimators,
	              'max_depth':max_depth,
	              'criterion': criterions
	              
	        }
	grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),
	                                 param_grid=parameters,
	                                 cv=cv,
	                                 n_jobs = -1)
	grid.fit(X_train,y_train) 
	# print (grid.best_score_)
	# print (grid.best_params_)
	# print (grid.best_estimator_)

	rf_grid = grid.best_estimator_
	rf_grid.score(X_test,y_test)

	rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
	                       criterion=grid.best_params_['criterion'], max_depth=grid.best_params_['max_depth'], 
	                       max_features='auto',
	                       max_leaf_nodes=None, max_samples=None,
	                       min_impurity_decrease=0.0, min_impurity_split=None,
	                       min_samples_leaf=1, min_samples_split=2,
	                       min_weight_fraction_leaf=0.0, n_estimators=grid.best_params_['n_estimators'],
	                       n_jobs=None, oob_score=False, random_state=None,
	                       verbose=0, warm_start=False)

	rf.fit(X_train, y_train)
	rfScore = round(rf.score(X_test, y_test),3)
	# print(rfScore)

	# # Print classification report for y_test
	rf_pred = rf.predict(X_test)
	# save model
	with open('model/randomForest.pickle', 'wb') as f:
		pickle.dump(rf, f)
	return rf_pred, rfScore 

def svmModel(X, y, X_train, X_test, y_train, y_test):
	# svm
	from sklearn.svm import SVC

	print("svm model processing")
	C = [1, 10, 100, 1000]
	gamma = [1e-3, 1e-4]
	kernel = ['rbf', 'linear', 'sigmoid']
	param_grid = {'C':C,
		              'gamma':gamma,
		              'kernel': kernel}

	## Calling on GridSearchCV object. 
	grid = GridSearchCV(estimator=SVC(), 
	                           param_grid = param_grid,
	                           cv = 4)
	
	from sklearn.preprocessing import MinMaxScaler
	scaling = MinMaxScaler(feature_range=(0,1)).fit(X_train)
	X_train = scaling.transform(X_train)
	X_test = scaling.transform(X_test)
	X = scaling.transform(X)
	## Fitting the model
	grid.fit(X, y)

	## Getting the best of everything. 
	# print (grid.best_score_)
	# print (grid.best_params_)
	# print(grid.best_estimator_)

	svcModel = SVC(C=grid.best_params_['C'], kernel=grid.best_params_['kernel'], degree=3, gamma='auto', 
					coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
					class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', 
	                random_state=55)

	svcModel.fit(X_train, y_train)
	svcScore = round(svcModel.score(X_test, y_test),3)
	# print(svcScore)

	# # Print classification report for y_test
	svc_pred = svcModel.predict(X_test)
	# save model
	with open('model/svc.pickle', 'wb') as f:
		pickle.dump(svcModel, f)
	return svc_pred, svcScore



if __name__ == '__main__':
	
	args = parse_arg()

	# prepare folder
	if not os.path.isdir("./model/"):
            os.mkdir("./model/")

	# read data
	X, y, X_train, X_test, y_train, y_test  = readData()
	# add label to the pic name
	xgb_pred, xgb_score = xgboostModel(X, y, X_train, X_test, y_train, y_test )
	knn_pred, knn_score = knnModel(X, y, X_train, X_test, y_train, y_test )
	gaussian_pred, gaussian_score = gaussianModel(X, y, X_train, X_test, y_train, y_test )
	lr_pred, lr_score= logisticRegressionModel(X, y, X_train, X_test, y_train, y_test )
	rf_pred, rf_score= randomForestModel(X, y, X_train, X_test, y_train, y_test )
	svc_pred, svc_score = svmModel(X, y, X_train, X_test, y_train, y_test )

	print("#####################################   xgboost model")
	print("accuracy: " + str(xgb_score))
	print(classification_report(y_test, xgb_pred))
	print(confusion_matrix(y_test,xgb_pred))

	print("#####################################   knn model")
	print("accuracy: " + str(knn_score))
	print(classification_report(y_test, knn_pred))
	print(confusion_matrix(y_test,knn_pred))

	print("#####################################   gaussian model")
	print("accuracy: " + str(gaussian_score))
	print(classification_report(y_test, gaussian_pred))
	print(confusion_matrix(y_test,gaussian_pred))

	print("#####################################   LogisticRegression 逻辑回归")
	print("accuracy: " + str(lr_score))
	print(classification_report(y_test, lr_pred))
	print(confusion_matrix(y_test,lr_pred))

	print("#####################################   random forest 随机森林")
	print("accuracy: " + str(rf_score))
	print(classification_report(y_test, rf_pred))
	print(confusion_matrix(y_test,rf_pred))

	print("#####################################   svm 支持向量机")
	print("accuracy: " + str(svc_score))
	print(classification_report(y_test, svc_pred))
	print(confusion_matrix(y_test,svc_pred))

	res = []
	final = svc_pred + rf_pred + lr_pred + gaussian_pred + knn_pred + xgb_pred

	for i in final:
	    if(i>2):
	        res.append(1)
	    else:
	        res.append(0)
	        
	res = np.array(res)
	# # Print classification report for y_test
	print("#####################################   voting result")
	print("accuracy: " + str(round(accuracy_score(y_test, res), 3)))
	print(classification_report(y_test, res))
	print(confusion_matrix(y_test,res)) 



	
