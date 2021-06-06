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

#substituimos todos os 0's da coluna prob e like para 1's
pima['prob']=pima['prob'].replace(0.0,1.0)
pima['like']=pima['like'].replace(0.0,1.0)


#o prob e like sao preenchido com a media
gf= pima['prob'].mean()
r= round(gf, 1)
pima['prob'].fillna(value=r,inplace=True)

gf= pima['like'].mean()
r=round(gf, 1)
pima['like'].fillna(value=r,inplace=True)

#os restantes com a moda exceto o age, age_o e id
gf= pima['met'].mode()[0]
pima['met'].fillna(value=gf,inplace=True)

gf= pima['length'].mode()[0]
pima['length'].fillna(value=gf,inplace=True)

gf= pima['int_corr'].mode()
pima['int_corr'].fillna(value=gf,inplace=True)

gf= pima['go_out'].mode()[0]
pima['go_out'].fillna(value=gf,inplace=True)

gf= pima['date'].mode()[0]
pima['date'].fillna(value=gf,inplace=True)

gf= pima['goal'].mode()[0]
pima['goal'].fillna(value=gf,inplace=True)

#para age e age_o faz-se a mediana
gf= pima['age_o'].median()
pima['age_o'].fillna(value=gf,inplace=True)

gf= pima['age'].median()
pima['age'].fillna(value=gf,inplace=True)

#aqui preenche o unico id a NaN por 22
pima['id'].fillna(value=22,inplace=True)


#convertemos de float para int
pima[['met','length','go_out','date','goal','age_o','age','id']]=  pima[['met','length','go_out','date','goal','age_o','age','id']].astype(int)
#dropamos a tabela int_corr pq é irrelevante neste momento
pima = pima.drop('int_corr',1)


#divisao do casos em q tem match e os q n tem match
resultados = pima['match']
dados = pima.drop(['match'],axis=1)


# Separa o nosso conjunto de dados em treino e teste (30%/70%)
dados_treino, dados_teste, resultados_treino, resultados_teste= train_test_split( dados, resultados, test_size=0.3,random_state=1)

#Chama o modelo Gaussian
gnb = GaussianNB()

#treina o modelo
gnb.fit(dados_treino,resultados_treino)

#dá uma previsão
predicted3 = gnb.predict(dados_teste)

print("Accuracy:",metrics.accuracy_score(resultados_teste, predicted3))

print("Matriz de confusão:\n",confusion_matrix(resultados_teste, gnb.predict(dados_teste)))

print("Classification Report:\n",classification_report(resultados_teste, gnb.predict(dados_teste)))