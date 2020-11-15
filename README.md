# Drug-Classification using Decision Tree with Classification And Regression Tree learning algorithm in GOlang
## Juanelv Salgado Sanchez - 2020

Teniendo como base un dataset provisto de la siguiente URL: https://www.kaggle.com/prathamtripathi/drug-classification

La clase o etiqueta del dataset es el tipo de droga y está basado en los siguientes features:
- Age
- Sex
- Blood Pressure Levels (BP)
- Cholesterol Levels
- Na to Potassium Ratio

El algoritmo de Machine Learning Supervisado utilizado es el Arbol de Decisiones

El Arbol de Decisiones es una técnica ampliamente utilizada en Machine Learning, puede ser utilizado para problemas de
clasificación y regresión. Su representación es un árbol binario, donde en cada nodo tiene una pregunta y esta sirve como un filtro
en donde divide los datos ingresados de acuerdo a si cumplen o no con la condición. La parte más importante de este algoritmo es la
manera en la que son elegidas estas preguntas y su orden. Para realizar dicha tarea y poder construir el árbol, se emplean diversos
algoritmos. El utilizado en este trabajo académico es el algoritmo CART (Classification And Regression Tree). Este algoritmo busca,
por cada feature o columna, iterar buscando los distintos valores y generar preguntas. Por cada una de las preguntas se genera un
indicador, en este caso es el Gini Impurity. Luego, a través de este indicador, se calcula la información ganada. Se selecciona la
pregunta con el índice de información ganada más alto, con el fin de dividr el dataset de la mejor manera. Finalmente, una vez
construido el árbol a partir de la data de entrenamiento, se procede ejecutar dicho árbol con la data de testing. Para este caso,
se obtuvo un porcentaje de precisión del 98%.

