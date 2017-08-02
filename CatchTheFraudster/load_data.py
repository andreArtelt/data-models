import pandas as pd


data_file = "FraudInstanceClean.csv"


def load_data():
	df = pd.read_csv(data_file, delimiter=',')

	y = df["Fraud.Instance"].copy().as_matrix()

	del df["Fraud.Instance"]
	del df["Unnamed: 0"]	# <=> Row count
	X = df.as_matrix()

	return X, y
