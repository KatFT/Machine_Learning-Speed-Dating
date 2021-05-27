import pandas as pd
import numpy as np
import sklearn 

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(dados_treino, resultados_treino)

#Predict the response for test dataset
predicted_test = clf.predict(dados_teste)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(resultados_teste, predicted_test))

print("Mean Absolute Error:",mae(resultados_teste,clf.predict(dados_teste)))

print("Mean Squared Error:",mse(resultados_teste,clf.predict(dados_teste)))

print(confusion_matrix(resultados_teste, clf.predict( dados_teste)))

print(classification_report(resultados_teste, clf.predict(dados_teste)))

#desenhar o grafo da arvore de decisao
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

feature_cols = ['id','partner','age','age_o','goal','date','go_out','length','met',	'like',	'prob','match']

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('id3.png')
Image(graph.create_png())
