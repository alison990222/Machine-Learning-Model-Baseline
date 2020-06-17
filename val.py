## python3 val.py --questionnair ml.csv --label label
import pandas as pd
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
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

	return X,y

def xgboostModel(X,y):
	print("xgboost model processing")
	with open('model/xgboost.pickle', 'rb') as f:
	    xgb = pickle.load(f)
	    #测试读取后的Model
	    xgb_pred = xgb.predict(X)
	    return xgb_pred

def knnModel(X,y):
	print("knn model processing")
	with open('model/knn.pickle', 'rb') as f:
	    knn = pickle.load(f)
	    #测试读取后的Model
	    knn_pred = knn.predict(X)
	    return knn_pred

def gaussianModel(X,y):
	print("gaussian model processing")
	with open('model/gaussian.pickle', 'rb') as f:
	    gaussian = pickle.load(f)
	    #测试读取后的Model
	    gaussian_pred = gaussian.predict(X)
	    return gaussian_pred

def logisticRegressionModel(X,y):
	print("LogisticRegression model processing")
	with open('model/LogisticRegression.pickle', 'rb') as f:
	    lr = pickle.load(f)
	    #测试读取后的Model
	    lr_pred = lr.predict(X)
	    return lr_pred

def randomForestModel(X,y):
	print("randomForest model processing")
	with open('model/randomForest.pickle', 'rb') as f:
	    rf = pickle.load(f)
	    #测试读取后的Model
	    rf_pred = rf.predict(X)
	    return rf_pred

def svmModel(X,y):
	print("svm model processing")
	with open('model/gaussian.pickle', 'rb') as f:
	    svm = pickle.load(f)
	    #测试读取后的Model
	    svm_pred = svm.predict(X)
	    return svm_pred

if __name__ == '__main__':
	
	args = parse_arg()
	# read data
	X, y = readData()
	# add label to the pic name

	xgb_pred = xgboostModel(X, y)
	knn_pred = knnModel(X, y)
	gaussian_pred = gaussianModel(X, y)
	lr_pred = logisticRegressionModel(X, y)
	rf_pred = randomForestModel(X, y)
	svc_pred = svmModel(X, y)

	print("#####################################   xgboost model")
	print("accuracy: " + str(round(accuracy_score(y, xgb_pred), 3)))
	print(classification_report(y, xgb_pred))
	print(confusion_matrix(y,xgb_pred))

	print("#####################################   knn model")
	print("accuracy: " + str(round(accuracy_score(y, knn_pred), 3)))
	print(classification_report(y, knn_pred))
	print(confusion_matrix(y,knn_pred))

	print("#####################################   gaussian model")
	print("accuracy: " + str(round(accuracy_score(y, gaussian_pred), 3)))
	print(classification_report(y, gaussian_pred))
	print(confusion_matrix(y,gaussian_pred))

	print("#####################################   LogisticRegression 逻辑回归")
	print("accuracy: " + str(round(accuracy_score(y, lr_pred), 3)))
	print(classification_report(y, lr_pred))
	print(confusion_matrix(y,lr_pred))

	print("#####################################   random forest 随机森林")
	print("accuracy: " + str(round(accuracy_score(y, rf_pred), 3)))
	print(classification_report(y, rf_pred))
	print(confusion_matrix(y,rf_pred))

	print("#####################################   svm 支持向量机")
	print("accuracy: " + str(round(accuracy_score(y, svc_pred), 3)))
	print(classification_report(y, svc_pred))
	print(confusion_matrix(y,svc_pred))

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
	print("accuracy: " + str(round(accuracy_score(y, res), 3)))
	print(classification_report(y, res))
	print(confusion_matrix(y,res)) 
