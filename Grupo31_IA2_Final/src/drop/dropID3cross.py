import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# load dataset
pima = pd.read_csv("speedDating_trab.csv")


pima=pima.dropna()

#convertemos de float para int
pima[['met','length','go_out','date','goal','age_o','age','id']]=  pima[['met','length','go_out','date','goal','age_o','age','id']].astype(int)


#divisao do cassos em q tem match e os q n tem match
resultados = pima['match']
dados = pima.drop(['match'],axis=1)


# Split dataset into training set and test set (30%/70%)
dados_treino, dados_teste, resultados_treino, resultados_teste= train_test_split( dados, resultados, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

#divide o nosso conjunto de dados em 10
kf = KFold(n_splits=10,shuffle=False)
kf.split(dados)

scores=cross_val_score(DecisionTreeClassifier(criterion="entropy"), dados, resultados, cv=10, scoring='accuracy')
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
    model = clf.fit(dados_treino, resultados_treino)
    # calcula a matriz de confusão
    score=confusion_matrix(resultados_teste, model.predict(dados_teste))
    print(score )
    c = score
    # soma as matrizes de confusões
    array = array + c
print("\n SOMA DA MATRIZ DE CONFUSÃO DE TODOS OS K-FOLD:\n",array,"\n")

#desenhar o grafo da arvore de decisao
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

feature_cols = ['id','partner','age','age_o','goal','date','go_out','length','met',	'like','int_corr','prob','match']

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dropID3cross.png')
Image(graph.create_png())
