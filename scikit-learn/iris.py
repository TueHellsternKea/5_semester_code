# Import nmodules
from sklearn.datasets import load_iris
from sklearn import tree

# Load Data
iris = load_iris()

# Use model
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# tree.plot_tree(clf)

import graphviz
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris")
