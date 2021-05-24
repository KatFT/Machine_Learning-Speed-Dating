import pandas as pd
import numpy as np
import sklearn 

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

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

gf= pima['id'].mode()[0]
pima['id'].fillna(value=gf,inplace=True)

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

