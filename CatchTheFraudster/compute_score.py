from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_score(y_true, y_pred):
	#return accuracy_score(y_true, y_pred)
	#return precision_score(y_true, y_pred)
	#return recall_score(y_true, y_pred)
	return f1_score(y_true, y_pred)

