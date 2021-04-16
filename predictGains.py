import pandas as pd
import matplotlib.pyplot as plt
import lstm
import matplotlib

predictedPrice={}
seq_len  = 500
readGains = False

def plotGains(strategies, numDays):
    gains = {}
    if readGains:
        gains = pd.read_csv('./out/gains.csv')
        gains['gain'] = gains['gain'].apply(lambda x: x * 1000 )

        minGain = gains.min()
        print('> minimum Gain : ', minGain)

        negs = gains.loc[gains.gain < 0].size

        if(negs < gains.size and negs > 0.0):
            gains['gain'] = gains['gain'].apply(lambda x: (x - minGain) )
        elif (negs == gains.size):
            gains['gain'] = gains['gain'].apply(lambda x: (x * (-1)) )
        else:
            print('All positive')

        plotStrategies(gains)
    else:

        for strategy in strategies:
            gains[strategy] = [calculateGains(strategies[strategy], numDays)]
            print('>  Gain: ', gains[strategy])

        df = pd.DataFrame.from_dict(gains, orient='columns')
        df.to_csv('./out/gains.csv', index=False)


def calculateGains(strategy, numDays):
    gain = 0.0
    for k in predictedPrice:
        gain = gain + float(predictedPrice[k][numDays-1])*float(strategy[k])
    return gain

def plotStrategies(df):
        font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 18}

        matplotlib.rc('font', **font)
        f, ax1 = plt.subplots(1, figsize=(10,10))

        bar_width = 0.5

        bar_l = [i+1 for i in range(len(df['MSFT']))]

        tick_pos = [i+(bar_width/2)-0.25 for i in bar_l]

        df['MSFT_tot'] = df['MSFT']* df['gain'] * 100
        df['GOOG_tot'] = df['GOOG']* df['gain'] * 100

        ax1.bar(bar_l, df['MSFT_tot'], width=bar_width, label='MSFT', alpha=0.8, color='#48B0F7')

        ax1.bar(bar_l, df['GOOG_tot'], width=bar_width, bottom=df['MSFT_tot'], label='GOOG', alpha=0.8, color='#F55753')


        plt.xticks(tick_pos, df['strategyName'])
        ax1.axes.yaxis.set_ticklabels([])

        ax1.set_ylabel("Rank")
        ax1.set_xlabel("Strategy")
        plt.legend(loc='upper right')

        plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])
        plt.savefig('./out/strategiesRank.png')

if __name__ == '__main__':
    strategies = {}
    strategies['st1']= {'GOOG':0.5,'MSFT' : 0.5}
    strategies['st2']={'GOOG' : 0.2, 'MSFT' :0.8}
    strategies['st3']= {'GOOG':0.9,'MSFT' : 0.1}
    strategies['st4']={'GOOG' : 0.4, 'MSFT' :0.6}

    if(readGains == False):
        print('> ** Predicting Google prices...')
        goog_prediction, goog_acc = lstm.calculate_price_movement('GOOG', seq_len)
        print('Google acc ' ,goog_acc )

        print('> ** Predicting Microsoft prices...')
        msft_prediction, msft_acc = lstm.calculate_price_movement('MSFT', seq_len)
        print('MSFT acc', msft_acc)

        msft_prediction_df = pd.DataFrame(msft_prediction)
        msft_prediction_df.to_csv('msft.csv', index=False)

        goog_prediction_df = pd.DataFrame(goog_prediction)
        goog_prediction_df.to_csv('goog.csv', index=False)

        print('> Including accuracy in predictions...')
        goog_prediction = [a * goog_acc for a in goog_prediction]
        msft_prediction = [a * msft_acc for a in msft_prediction]

        predictedPrice['GOOG'] = goog_prediction
        predictedPrice['MSFT'] = msft_prediction

    plotGains(strategies, seq_len)