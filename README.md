# Previsão de Aprovação de Empréstimos com Árvores de Decisão
**Introdução:** Este repositório contém código para prever aprovações de empréstimos usando classificadores de árvore de decisão. O conjunto de dados utilizado neste projeto é chamado de 'demostracao.csv', que inclui vários recursos relacionados às informações financeiras e pessoais dos indivíduos. O objetivo é construir um modelo que possa prever com precisão se um pedido de empréstimo deve ser aprovado ou não.

# Visão geral do código:
**Preparação de dados:** 
O código começa importando as bibliotecas necessárias: pandas, numpy e seaborn...
```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus
from sklearn.metrics import confusion_matrix
```

O arquivo 'demostracao.csv' é carregado em um DataFrame pandas chamado 'demo'.
```python
demo = pd.read_csv('/content/demostracao.csv')
demo.head(5)
```
![Captura de tela 2023-05-19 151802](https://github.com/TKlebis/Credito-Bancario/assets/130613291/96f0fb19-85e1-4dfb-acae-538ab35d54a6)

**Engenharia de recursos:** Um novo DataFrame chamado 'demo_info' é criado para armazenar informações sobre as variáveis ​​do conjunto de dados e seus tipos de dados.

```python
demo_info = pd.DataFrame({'variavel': demo.columns, 'tipo': demo.dtypes})

qtd_categorias = []
for var in demo.columns:
    qtd_categorias.append(demo[var].nunique())
demo_info['qtd_categorias'] = qtd_categorias


demo = pd.get_dummies(demo, columns=['tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia'])


demo_info
demo.head()
```
O código calcula o número de categorias exclusivas para cada variável e adiciona essas informações a 'demo_info'.<br>
A codificação one-hot é aplicada a variáveis ​​categóricas usando a função 'get_dummies', resultando em um DataFrame 'demo' atualizado.

**Visualização de dados:** O código seleciona recursos numéricos de 'demo' e cria um DataFrame chamado 'x' contendo apenas essas colunas.
Uma tabulação cruzada e um mapa de calor são gerados para visualizar a relação entre propriedade de e-mail, propriedade de veículo e taxas de inadimplência de empréstimos.

```python
numer = demo.select_dtypes(include=['float64', 'int64', 'uint8'])
x = pd.concat([numer], axis=1)
x.head()
```
![Captura de tela 2023-05-19 152644](https://github.com/TKlebis/Credito-Bancario/assets/130613291/61a64c7a-5084-4f95-a12c-03ae52a5818e)

```python
pd.crosstab(demo.possui_email, demo.posse_de_veiculo, values=demo.mau, aggfunc=np.mean)


sns.heatmap(pd.crosstab(demo.possui_email, demo.posse_de_veiculo, values=demo.mau, aggfunc=np.mean), annot=True, cmap='Blues')
```

**Treinamento modelo:** O conjunto de dados é dividido em conjuntos de treinamento e validação usando a função 'train_test_split'.
Um classificador de árvore de decisão é instanciado.
A codificação one-hot é aplicada a variáveis ​​categóricas adicionais em 'demo'.
O classificador da árvore de decisão é ajustado aos dados de treinamento.

```python 
x_treino, x_valid, y_treino, y_valid = train_test_split(demo.drop('mau', axis=1), demo['mau'], test_size=0.3, random_state=42)
arvore = DecisionTreeClassifier()
demo = pd.get_dummies(demo, columns=['possui_email', 'posse_de_veiculo', 'posse_de_imovel'])
demo = pd.get_dummies(demo, columns=['sexo'])
arvore.fit(x_treino, y_treino)
```
**Avaliação do modelo:** A árvore de decisão é visualizada usando a função 'export_graphviz' e exibida como uma imagem.
As previsões são feitas no conjunto de validação e uma matriz de confusão é gerada para avaliar o desempenho do modelo.
A precisão do modelo no conjunto de treinamento é calculada e impressa.
Métricas de avaliação adicionais, como precisão no conjunto de validação, são calculadas e exibidas.

```python
d = export_graphviz(arvore, out_file=None,
                           feature_names=x_treino.columns,
                           class_names=['Reprovado', 'Aprovado'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = pydotplus.graph_from_dot_data(d)
Image(graph.create_png())
```
![Captura de tela 2023-05-19 153531](https://github.com/TKlebis/Credito-Bancario/assets/130613291/9faa6551-309f-41b1-9f0a-3015730296f4)

```python
y_pred = arvore.predict(x_valid)
labels = ['Reprovado', 'Aprovado']
cm = confusion_matrix(y_valid, y_pred)
demoo = pd.DataFrame(cm, index=labels, columns=labels)
demoo.index.name = 'Real'
demoo.columns.name = 'Previsto'
print(demoo)
```
![Captura de tela 2023-05-19 153640](https://github.com/TKlebis/Credito-Bancario/assets/130613291/d4f0f996-699a-4187-8d25-438b6d6e1489)

```python
acuracia = arvore.score(x_treino, y_treino)
print("A acurácia é:", acuracia)
```
A acurácia é: 0.9912483912483913

**Otimização do modelo:** Um novo classificador de árvore de decisão é criado com hiperparâmetros específicos (random_state, min_samples_leaf, max_depth).
O modelo otimizado é treinado no conjunto de treinamento e avaliado no conjunto de validação usando uma matriz de confusão.
A porcentagem de solicitações de empréstimo 'boas' (aprovadas) no conjunto de validação é calculada e impressa.

```python
teste = arvore.predict(x_valid)
confusao_teste = confusion_matrix(y_valid, teste)
print(confusao_teste)
from sklearn.metrics import accuracy_score

acuracia_valid = accuracy_score(y_valid, teste)
print(acuracia_valid)
```
0.9683683683683684

```python
predicao_trei = arvore.predict(x_treino)
acuracia_trei = accuracy_score(y_treino, predicao_trei)
print(acuracia_trei)
```
0.9912483912483913

```python
arvore2 = DecisionTreeClassifier(random_state=123, min_samples_leaf=5, max_depth=10)
arvore2.fit(x_treino, y_treino)
predicao_teste2 = arvore2.predict(x_valid)
confusao_teste2 = confusion_matrix(y_valid, predicao_teste2)
print(confusao_teste2)

[[4862   25]
 [ 100    8]]

pd.Series(predicao_teste2).value_counts()
False    4962
True       33
dtype: int64

bons = y_valid.value_counts()[0] / len(y_valid)
print(bons)
0.9783783783783784
```

## Conclusão: 

Este código demonstra o uso de classificadores de árvore de decisão para prever aprovações de empréstimos com base em vários atributos do cliente. O modelo de árvore de decisão é treinado em um conjunto de dados fornecido e seu desempenho é avaliado usando matrizes de precisão e confusão. Além disso, um modelo de árvore de decisão otimizado é criado por hiperparâmetros de ajuste fino para melhorar potencialmente a precisão da previsão. O código e a documentação que o acompanha podem servir como ponto de partida para exploração e aprimoramento adicionais de modelos de previsão de aprovação de empréstimos usando árvores de decisão.



