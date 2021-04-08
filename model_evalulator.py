'''
import model_evalulator as me
import numpy as np


actual = np.array(['NonDemented', 'NonDemented', 'MildDemented', 'NonDemented',
       'MildDemented', 'NonDemented', 'VeryMildDemented', 'NonDemented',
       'VeryMildDemented', 'VeryMildDemented', 'NonDemented',
       'MildDemented', 'NonDemented', 'NonDemented', 'NonDemented',
       'VeryMildDemented', 'MildDemented', 'NonDemented',
       'VeryMildDemented', 'NonDemented', 'NonDemented', 'MildDemented',
       'NonDemented', 'NonDemented', 'NonDemented', 'NonDemented',
       'NonDemented', 'VeryMildDemented', 'NonDemented',
       'VeryMildDemented', 'NonDemented', 'VeryMildDemented',
       'MildDemented', 'NonDemented', 'MildDemented', 'NonDemented',
       'MildDemented', 'MildDemented', 'NonDemented', 'NonDemented',
       'VeryMildDemented', 'NonDemented', 'MildDemented', 'MildDemented',
       'NonDemented', 'VeryMildDemented', 'NonDemented', 'NonDemented',
       'VeryMildDemented', 'MildDemented'])

predicted = np.array(['NonDemented', 'NonDemented', 'MildDemented', 'NonDemented',
       'MildDemented', 'VeryMildDemented', 'VeryMildDemented',
       'NonDemented', 'VeryMildDemented', 'VeryMildDemented',
       'NonDemented', 'MildDemented', 'NonDemented', 'NonDemented',
       'NonDemented', 'VeryMildDemented', 'MildDemented', 'NonDemented',
       'VeryMildDemented', 'NonDemented', 'VeryMildDemented',
       'MildDemented', 'NonDemented', 'NonDemented', 'NonDemented',
       'NonDemented', 'NonDemented', 'VeryMildDemented', 'NonDemented',
       'VeryMildDemented', 'NonDemented', 'VeryMildDemented',
       'MildDemented', 'NonDemented', 'MildDemented', 'NonDemented',
       'MildDemented', 'MildDemented', 'NonDemented', 'NonDemented',
       'VeryMildDemented', 'NonDemented', 'MildDemented', 'MildDemented',
       'NonDemented', 'VeryMildDemented', 'NonDemented', 'NonDemented',
       'VeryMildDemented', 'MildDemented'])

e = me.Evalulator(actual, predicted)

cm = e.confusion_matrix(True)

'''


import pandas as pd
import sklearn.metrics as skm
import seaborn as sns
import matplotlib.pyplot as plt

class Evalulator(object):
	def __init__(self, true_labels='', prediction_labels=''):
		if hasattr(true_labels, '__len__') and not isinstance(true_labels, str) and len(true_labels) > 0:
			self.actual = true_labels
		else:
			raise Exception('The test data provided either does not exist or did not come in the necessary format.')

		if hasattr(prediction_labels, '__len__') and not isinstance(prediction_labels, str) and len(prediction_labels) > 0:
			self.predicted = prediction_labels
		else:
			raise Exception('The prediction data provided either does not exist or did not come in the necessary format.')

	def confusion_matrix(self, show=False):
		results = {'actual':self.actual, 'predicted':self.predicted}
		confusion_matrix = pd.crosstab(results['actual'], results['predicted'], rownames=['Actual'], colnames=['Predicted'])

		cm = sns.heatmap(confusion_matrix, annot=True, linewidths=.5, cmap=sns.cubehelix_palette(8))
		
		if show == True:
			plt.show()
		else:
			return cm

	def stats(self):
		print('Accuracy: %.5f' % skm.accuracy_score(self.actual, self.predicted))
		#print('Balanced Accuracy: %.5f' % skm.balanced_accuracy_score(self.actual, self.predicted))
		print('F1: %.5f' % skm.f1_score(self.actual, self.predicted, average='weighted'))
		print('Recall: %.5f' % skm.recall_score(self.actual, self.predicted, average='weighted'))
		print('Precision: %.5f' % skm.precision_score(self.actual, self.predicted, average='weighted'))
		print('Jaccard: %.5f' % skm.jaccard_score(self.actual, self.predicted, average='weighted'))