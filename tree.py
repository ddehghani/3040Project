import pandas as pd
import numpy as np
from scipy.stats import entropy
from graphviz import Digraph

class DesicionTreeNode: #a class to represent a node is the desicion tree
	def __init__(self, information = 0, value = '', next_branch_variable = '', samples = 0):
		self.information = information
		self.value = value
		self.next_branch_variable = next_branch_variable
		self.children = []
		self.samples = samples
		self.isLeafNode = False
		self.leafNodeValue = ''
	def __repr__(self):
		return f'Value= {self.value}'\
		f'\nSamples= {self.samples}'\
		f'\nInfo= {self.information}'\
		f'\nNext Branch= {self.next_branch_variable}'\
		f'\nChildren= {len(self.children)}'\
		'\n' + f'{"class=" + self.leafNodeValue if self.isLeafNode else ""}'

def buildDecisionTree(data, classAttribute, root=None): #recursive function to build desicion tree
	#calculate the entropy and the number of instances
	root_info = entropy(data[classAttribute].value_counts(), base=2)
	data_count = data.shape[0]
	
	#if the entropy is 0 or we have no more data or we have only one attribute class is the majority
	if root_info == 0 or data_count == 0 or data.shape[1] == 1:
		root.isLeafNode = True
		root.leafNodeValue = data[classAttribute].value_counts().idxmax()
		return

	info = {}
	#for each attribute in data
	for attribute in data.drop([classAttribute], axis=1).keys():
		info[attribute] = 0
		#for each unique value in that variable
		for unique_value in data[attribute].unique():
			data_i = data[data[attribute] == unique_value]
			info_i = entropy(data_i[classAttribute].value_counts(), base=2)
			p_i = data_i.shape[0] / data_count
			info[attribute] += p_i * info_i
	#choose the variable that results in minimum entropy(info) or maximum information gain!
	next_branch_variable = min(info, key=info.get)
	
	if root == None:
		root = DesicionTreeNode(
			information = root_info, 
			value = 'root', 
			next_branch_variable = next_branch_variable, 
			samples = data_count)
	else:
		root.information = root_info
		root.next_branch_variable = next_branch_variable

	for unique_value in data[next_branch_variable].unique():
		newData = data[data[next_branch_variable] == unique_value].drop(columns=[next_branch_variable],axis=1)
		childNode = DesicionTreeNode(value = unique_value, samples = newData.shape[0])
		buildDecisionTree(newData, classAttribute, childNode)
		root.children.append(childNode)
	return root

def buildGraph(root, graph = None): #recursively traverse the tree and build a visual graph
	if graph == None:
		graph = Digraph('G', filename='dt.gv', node_attr={'shape': 'record'})
	for child in root.children:
		graph.edge(str(root), str(child))
		buildGraph(child, graph)
	return graph

def classify(root, dataPoint):
	if root.isLeafNode:
		return root.leafNodeValue
	dataValue = dataPoint[root.next_branch_variable]
	for child in root.children:
		if child.value == dataValue:
			return classify(child, dataPoint)

def evaluateTree(root, testData, classAttribute):
	success = 0
	for index, data in testData.iterrows():
		if data[classAttribute] == classify(root, data):
			success += 1
	return success/testData.shape[0]