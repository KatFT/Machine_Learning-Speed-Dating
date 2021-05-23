import pandas as pd
import numpy as np
import sklearn 
# Importing the statistics module
import statistics

#from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier


# load dataset
pima = pd.read_csv("speedDating_trab.csv")
#isto vai ter q sair (marosca do duarte)
#pima= pima.dropna()
#pima.head()



#para encher valores nan podemos utilizar o fillna()
pima['prob'].fillna(pima.mean())


#divisao do cassos em q tem match e os q n tem match
resultados = pima['match']
dados = pima.drop(['match'],axis=1)

# Split dataset into training set and test set
dados_treino, dados_teste, resultados_treino, resultados_teste= train_test_split( dados, resultados, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(dados_treino, resultados_treino)

#Predict the response for test dataset
predricted = clf.predict(dados_teste)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(resultados_teste, predricted))