

## Questionnair Train and Val

#### 训练:  train.py

`python3 train.py --questionnair` ml.csv (data csv) ` --label` label (label column)

(`python3 change_label.py -h` to see more)



#### 训练 + 保存: trainAndSave.py

`python3 trainAndSave.py --questionnair` ml.csv (data csv) ` --label` label (label column)

> Output: model 文件夹下会有六个权重模型(.pickle文件)



#### 预测:  val.py

`python3 val.py --questionnair` ml.csv (data csv) ` --label` label (label column)



#### 打印数据

会依次打印 xgboost, knn, gaussian, logistic regression, random forest, Svm, 组合voting 等七种模型数据

<img src="./pic/data.png" alt="Screen Shot 2020-06-17 at 5.46.35 PM" width="380" height="200" />