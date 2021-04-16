import lstm
import dataRetrieval
import time
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

predictedPrice={}
seq_len  = 500
readGains = False

def plot_results(predicted_data, true_data, fileName):
	fig = plt.figure(facecolor='white', figsize=(20,10))
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	plt.plot(predicted_data, label='Prediction')
	plt.legend()
	plt.savefig(fileName)

def plot_results_multiple(predicted_data, true_data, prediction_len):
	fig = plt.figure(facecolor='white', figsize=(30,10))
	ax = fig.add_subplot(111)
	ax.plot(true_data, label='True Data')
	for i, data in enumerate(predicted_data):
		padding = [None for p in range(i * prediction_len)]
		plt.plot(padding + data, label='Prediction')
		plt.legend()
		plt.savefig('./out/multipleResults.jpg')

def plotMetrics(history):
	losses = []
	mses = []
	for key, value in history.items():
		if(key == 'loss'):
			losses = value
	plt.figure(figsize=(6, 3))
	plt.plot(losses)
	plt.ylabel('error')
	plt.xlabel('iteration')
	plt.title('testing error over time')
	plt.savefig('losses.png')

def trainModel(newModel):
	if newModel:
		global_start_time = time.time()

		print('> Data Loaded. Compiling LSTM model...')

		model = lstm.build_model()

		model.save('./model/lstm.h5')

		print('> Training duration (s) : ', time.time() - global_start_time)
	else:
		print('> Data Loaded. Loading LSTM model...')

		model = load_model('./model/lstm.h5')

	return model

def run():
	dataRetrieval.retrieveStockData('AMZN', '2015-05-15', '2020-05-15', './data/lstm/AMZN.csv')
	dataRetrieval.retrieveStockData('GOOG', '2015-05-15', '2020-05-15', './data/lstm/GOOG.csv')
	dataRetrieval.retrieveStockData('IBM', '2015-05-15', '2020-05-15', './data/lstm/IBM.csv')
	dataRetrieval.retrieveStockData('MSFT', '2015-05-15', '2020-05-15', './data/lstm/MSFT.csv')

	stockFile = 'C:\\Users\\aaron\\PycharmProjects\\pythonProject\\data\\lstm\\GOOG.csv'
	epochs = 11
	seq_len = 50
	batch_size= 100

	print('> Loading data... ')

	X_train, y_train, X_test, y_test = lstm.load_data(stockFile, seq_len, True)

	X_train = pad_sequences(X_train, maxlen=seq_len)
	X_test = pad_sequences(X_test, maxlen=seq_len)

	print('> X_train seq shape: ', X_train.shape)
	print('> X_test seq shape: ', X_test.shape)

	model = trainModel(True)

	print('> LSTM trained, Testing model on validation set... ')

	training_start_time = time.time()

	hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.05, validation_data=(X_test, y_test))

	print('> Testing duration (s) : ', time.time() - training_start_time)

	score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
	print('Test score:', score)
	print('Test accuracy:', acc)

	print('> ** Predicting Google prices...')
	goog_prediction, goog_acc = lstm.calculate_price_movement('GOOG', seq_len)
	print('Google acc ', goog_acc)

	# print('> ** Predicting Microsoft prices...')
	# msft_prediction, msft_acc = lstm.calculate_price_movement('MSFT', seq_len)
	# print('MSFT acc', msft_acc)

	# msft_prediction_df = pd.DataFrame(msft_prediction)
	# msft_prediction_df.to_csv('msft.csv', index=False)

	goog_prediction_df = pd.DataFrame(goog_prediction)
	goog_prediction_df.to_csv('goog.csv', index=False)

	print('> Including accuracy in predictions...')
	goog_prediction = [a * goog_acc for a in goog_prediction]
	# msft_prediction = [a * msft_acc for a in msft_prediction]

	predictedPrice['GOOG'] = goog_prediction
	# predictedPrice['MSFT'] = msft_prediction

	print('> Plotting Losses....')

	print('> Plotting point by point prediction....')
	predicted = lstm.predict_point_by_point(model, X_test)

	plot_results(predicted, y_test, './out/ppResults.jpg')

	print('> Plotting full sequence prediction....')
	predicted = lstm.predict_sequence_full(model, X_test, seq_len)
	plot_results(predicted, y_test, './out/sResults.jpg')

	print('> Plotting multiple sequence prediction....')
	predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
	plot_results_multiple(predictions, y_test, 50)

if __name__=='__main__':
	run()