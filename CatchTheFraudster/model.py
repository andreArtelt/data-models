from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from load_data import load_data
from compute_score import compute_score



if __name__ == "__main__":
	X, y = load_data()

	kf = KFold(n_splits=10, shuffle=True)
	for train_idx, test_idx in kf.split(X):
		x_train, y_train, x_test, y_test = X[train_idx], y[train_idx], X[test_idx], y[test_idx]

		#model = LogisticRegression(C=0.1)
		model = DecisionTreeClassifier(max_depth=4)
		model.fit(x_train, y_train)
		#print model.feature_importances_

		y_train_pred = model.predict(x_train)
		y_test_pred = model.predict(x_test)

		print compute_score(y_train, y_train_pred)
		print compute_score(y_test, y_test_pred)
		print ""
