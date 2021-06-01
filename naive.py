from math import gamma
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# load dataset
pima = pd.read_csv("speedDating_trab.csv")

#tds os nans de tds as colunas menos o int_corr vamos preencher com a moda
#o int_corr preenchemos com a media

gf= pima['prob'].mode()[0]
pima['prob'].fillna(value=gf,inplace=True)

gf= pima['like'].mode()[0]
pima['like'].fillna(value=gf,inplace=True)

gf= pima['met'].mode()[0]
pima['met'].fillna(value=gf,inplace=True)

gf= pima['length'].mode()[0]
pima['length'].fillna(value=gf,inplace=True)

gf= pima['int_corr'].mean()
pima['int_corr'].fillna(value=gf,inplace=True)

gf= pima['go_out'].mode()[0]
pima['go_out'].fillna(value=gf,inplace=True)

gf= pima['date'].mode()[0]
pima['date'].fillna(value=gf,inplace=True)

gf= pima['goal'].mode()[0]
pima['goal'].fillna(value=gf,inplace=True)

gf= pima['age_o'].mode()[0]
pima['age_o'].fillna(value=gf,inplace=True)

gf= pima['age'].mode()[0]
pima['age'].fillna(value=gf,inplace=True)

#gf= pima['id'].mode()[0]
#pima['id'].fillna(value=gf,inplace=True)
pima.dropna(subset=['id'],inplace=True )

#convertemos de float para int
pima[['prob','like','met','length','go_out','date','goal','age_o','age','id']]=  pima[['prob','like','met','length','go_out','date','goal','age_o','age','id']].astype(int)

#dropamos a tabela int_corr pq é irrelevante neste momento
pima = pima.drop('int_corr',1)

#divisao do cassos em q tem match e os q n tem match
resultados = pima['match']
dados = pima.drop(['match'],axis=1)


# Split dataset into training set and test set (30%/70%)
dados_treino, dados_teste, resultados_treino, resultados_teste= train_test_split( dados, resultados, test_size=0.3, random_state=1)


#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(dados_treino,resultados_treino)

#Predict the response for test dataset
predicted = gnb.predict(dados_teste)


print("Accuracy:",metrics.accuracy_score(resultados_teste, predicted))

#print("Mean Absolute Error:",mae(resultados_teste,gnb.predict(dados_teste)))

#print("Mean Squared Error:",mse(resultados_teste,gnb.predict(dados_teste),squared=False))

print(confusion_matrix(resultados_teste, gnb.predict( dados_teste)))

print(classification_report(resultados_teste, gnb.predict(dados_teste)))

kf = KFold(n_splits=7,shuffle=False)
kf.split(dados)

scores=cross_val_score(GaussianNB(), dados, resultados, cv=7, scoring='accuracy')
print("Cross Validation:", scores)
print("The mean value for K-fold cross validation test that best explains our model is {}".format(scores.mean()),"\n")

sizes, training_scores, testing_scores = learning_curve(GaussianNB(), dados, resultados, cv=10, scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 50))
  
# Mean and Standard Deviation of training scores
mean_training = np.mean(training_scores, axis=1)
Standard_Deviation_training = np.std(training_scores, axis=1)
  
# Mean and Standard Deviation of testing scores
mean_testing = np.mean(testing_scores, axis=1)
Standard_Deviation_testing = np.std(testing_scores, axis=1)
  
# dotted blue line is for training scores and green line is for cross-validation score
plt.plot(sizes, mean_training, '--', color="b",  label="Training score")
plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")
  
# Drawing plot
plt.title("LEARNING CURVE FOR GAUSSIAN")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


