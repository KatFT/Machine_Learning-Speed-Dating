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

#divisao do cassos em q tem match e os q n tem match
resultados = pima['match']
dados = pima.drop(['match'],axis=1)



# Split dataset into training set and test set (30%/70%)
dados_treino, dados_teste, resultados_treino, resultados_teste= train_test_split( dados, resultados, test_size=0.3, random_state=1)


#Chama o modelo Gaussian
gnb = GaussianNB()

#divide o nosso conjunto de dados em 10
kf = KFold(n_splits=10,shuffle=False)
kf.split(dados)

scores=cross_val_score(GaussianNB(), dados, resultados, cv=10, scoring='accuracy')
print("Cross Validation:\n", scores)
print("A média para Cross-Validation do K-fold é: {}".format(scores.mean()),"\n")

# Inicializa o array a zero que vai guardar a nossa matriz de confusão
array = [[0,0],[0,0]]
print("MATRIZ DE CONFUSÃO DE CADA K-FOLD:")
# para cada split train test vai treinar, prever e fazer a matriz de confusão
for train_index, test_index in kf.split(dados):
    # split train test
    dados_treino, dados_teste = dados.iloc[train_index], dados.iloc[test_index]
    resultados_treino, resultados_teste = resultados.iloc[train_index], resultados.iloc[test_index]
     # treina o modelo
    model = gnb.fit(dados_treino, resultados_treino)
    # calcula a matriz de confusão
    score=confusion_matrix(resultados_teste, model.predict(dados_teste))
    print(score )
    c = score
    # soma as matrizes de confusões
    array = array + c
print("\n SOMA DA MATRIZ DE CONFUSÃO DE TODOS OS K-FOLD:\n",array,"\n")

# cv é o número de folds enquanto faz Cross-Validation
sizes, training_scores, testing_scores = learning_curve(GaussianNB(), dados, resultados, cv=10, scoring='accuracy', train_sizes=np.linspace(0.01, 1.0, 50))
  
# Média e desvio padrão do score do treino
mean_training = np.mean(training_scores, axis=1)
Standard_Deviation_training = np.std(training_scores, axis=1)
  
# Média e desvio padrão do score do teste
mean_testing = np.mean(testing_scores, axis=1)
Standard_Deviation_testing = np.std(testing_scores, axis=1)
  
# azul trajecado é para os valores de treino e a linha verde para o valor cross.-validation
plt.plot(sizes, mean_training, '--', color="b",  label="Training score")
plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")
  
# Desenha gráfico
plt.title("LEARNING CURVE FOR GAUSSIAN")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()