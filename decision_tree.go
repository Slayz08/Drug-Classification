/*
Drug Classifcation using Decision Tree with Classification And Regression Tree learning algorithm in GOlang
Juanelv Salgado Sanchez - 2020

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


*/

package main

import (
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/go-gota/gota/series"

	"github.com/go-gota/gota/dataframe"
)

type Question struct {
	columnName     string
	value          string
	isNumericValue bool
}

type Node struct {
	nodeType    string //Leaf or Decision
	question    Question
	trueBranch  *Node
	falseBranch *Node
	predictions map[string]int
}

func newDecisionNode(question Question, trueBranch Node, falseBranch Node) Node {
	return Node{"decision", question, &trueBranch, &falseBranch, nil}
}

func newLeafNode(df dataframe.DataFrame) Node {
	predictions := countLabels(df)
	return Node{"leaf", Question{}, &Node{}, &Node{}, predictions}
}

func checkQuestion(q Question, row dataframe.DataFrame) bool {
	value := row.Col(q.columnName).Records()[0]

	if q.isNumericValue {
		_value, _ := strconv.ParseFloat(value, 8)
		comparisonValue, _ := strconv.ParseFloat(q.value, 8)
		return _value <= comparisonValue
	} else {
		return value == q.value
	}

}

func loadDataset(file string) dataframe.DataFrame {

	csvfile, err := os.Open(file)

	if err != nil {
		log.Fatal(err)

	}

	//drugTrainingDataSet := qframe.ReadCSV(csvfile)
	//fmt.Println(drugTrainingDataSet)

	df := dataframe.ReadCSV(csvfile)

	return df

}

//get unique values from a string slice
func getUniques(list []string) (set []string) {
	ks := make(map[string]bool) // map to keep track of repeats

	for _, e := range list {
		//if the element is not present on map, then create it
		if _, v := ks[e]; !v {
			ks[e] = true
			set = append(set, e)
		}
	}
	return
}

func countLabels(df dataframe.DataFrame) map[string]int {
	labelsCounter := make(map[string]int)

	drugColSeries := df.Col("Drug")
	distinctLabels := getUniques(drugColSeries.Records())

	//initializing the map
	for i := 0; i < len(distinctLabels); i++ {
		labelsCounter[distinctLabels[i]] = 0
	}

	for i := 0; i < drugColSeries.Len(); i++ {
		currentLabel := drugColSeries.Elem(i).String()
		labelsCounter[currentLabel] += 1
	}

	//fmt.Println(df.Col("Drug"))
	//fmt.Println(distinctLabels)
	//fmt.Println(labelsCounter)
	return labelsCounter
}

//Calculate Gini Impurity value
func calcGini(df dataframe.DataFrame) float64 {
	labelsCount := countLabels(df)
	var impurity float64 = 1.00

	for _, count := range labelsCount {
		var prob float64 = float64(count) / float64(df.Nrow())
		impurity -= prob * prob
	}
	return impurity
}

func generateBranches(df dataframe.DataFrame, question Question) (dataframe.DataFrame, dataframe.DataFrame) {

	slicesAux := make([][]string, 1)
	slicesAux[0] = df.Names()
	trueBranch := dataframe.New(
		series.New([]string{}, series.Int, "Age"),
		series.New([]string{}, series.String, "Sex"),
		series.New([]string{}, series.String, "BP"),
		series.New([]string{}, series.String, "Cholesterol"),
		series.New([]string{}, series.Float, "Na_to_K"),
		series.New([]string{}, series.String, "Drug"),
	)
	falseBranch := dataframe.New(
		series.New([]string{}, series.Int, "Age"),
		series.New([]string{}, series.String, "Sex"),
		series.New([]string{}, series.String, "BP"),
		series.New([]string{}, series.String, "Cholesterol"),
		series.New([]string{}, series.Float, "Na_to_K"),
		series.New([]string{}, series.String, "Drug"),
	)

	for row := 0; row < df.Nrow(); row++ {
		//add to the respective df, according to the question
		if checkQuestion(question, df.Subset(row)) {
			trueBranch = trueBranch.RBind(df.Subset(row))
		} else {
			falseBranch = falseBranch.RBind(df.Subset(row))
		}

	}
	return trueBranch, falseBranch
}

func calcInfoGain(trueBranch dataframe.DataFrame, falseBranch dataframe.DataFrame, currentUncertainty float64) float64 {
	p := float64(trueBranch.Nrow()) / float64(trueBranch.Nrow()+falseBranch.Nrow())
	return currentUncertainty - p*calcGini(trueBranch) - (1-p)*calcGini(falseBranch)
}

//Returns the gain and the question (represented by index of column)
func findBestQuestion(df dataframe.DataFrame) (gain float64, question Question) {
	maxGain := 0.00
	currentUncertainty := calcGini(df)
	columnNames := df.Names()
	bestQuestion := Question{"asdad", "Easda", false}

	for _, col := range columnNames {

		if col == "Drug" {
			continue
		}

		//get unique values in column
		uniqueValues := getUniques(df.Col(col).Records())

		for _, val := range uniqueValues {
			//create a question for each distinct value in a column
			//fmt.Printf("Col: %s, value: %s\n", col, val)

			//defining text or numeric values
			isNumericValue := false
			if col == "Age" || col == "Na_to_K" {
				isNumericValue = true
			}
			question := Question{col, val, isNumericValue}

			//separating dataset in two branches
			trueBranch, falseBranch := generateBranches(df, question)

			if trueBranch.Nrow() == 0 || falseBranch.Nrow() == 0 {
				continue
			}

			gain := calcInfoGain(trueBranch, falseBranch, currentUncertainty)

			if gain >= maxGain {
				bestQuestion = question
				maxGain = gain
			}

		}

	}

	return maxGain, bestQuestion
}

func generateTree(df dataframe.DataFrame) Node {
	gain, question := findBestQuestion(df)

	//if there's no info gain, we're done
	if gain == 0 {
		return newLeafNode(df)
	}

	trueBranchData, falseBranchData := generateBranches(df, question)

	trueBranchNode := generateTree(trueBranchData)
	falseBranchNode := generateTree(falseBranchData)

	decisionNode := newDecisionNode(question, trueBranchNode, falseBranchNode)
	return decisionNode

}

func printLeafNode(labelsCount map[string]int) map[string]string {

	total := 0
	for _, count := range labelsCount {
		total += count
	}
	total = total * 1.0

	probs := make(map[string]string)
	for label, count := range labelsCount {
		probs[label] = fmt.Sprintf("%f %%", float64(count)/float64(total)*100)
	}

	return probs
}

func classify(row dataframe.DataFrame, node *Node) map[string]string {
	if node.nodeType == "leaf" {
		//fmt.Println(node.predictions)
		//return dataframe.New()
		return printLeafNode(node.predictions)
	}

	//classify with the child decision node
	if checkQuestion(node.question, row) {
		return classify(row, node.trueBranch)
	} else {
		return classify(row, node.falseBranch)
	}
}

func main() {
	drugTrainingDataSet := loadDataset("C:/Users/slayz.DESKTOP-9QQK0G7/OneDrive - Universidad Peruana de Ciencias/2020-2/Programación Concurrente/TB2/drugsTraining.csv")
	drugTestingDataSet := loadDataset("C:/Users/slayz.DESKTOP-9QQK0G7/OneDrive - Universidad Peruana de Ciencias/2020-2/Programación Concurrente/TB2/drugsTest.csv")
	//fmt.Println(drugTrainingDataSet)

	decisionTree := generateTree(drugTrainingDataSet)

	//fmt.Println(calcGini(drugTrainingDataSet))
	//findBestQuestion(drugTrainingDataSet)
	for row := 0; row < drugTestingDataSet.Nrow(); row++ {
		fmt.Printf("Actual: %s - Predicted: %s \n", drugTestingDataSet.Elem(row, 5), classify(drugTestingDataSet.Subset(row), &decisionTree))
	}
}
