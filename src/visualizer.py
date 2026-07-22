import matplotlib.pyplot as plt


def plot_features(df):
    
    fig, ax = plt.subplots(3, 1, figsize=(14, 10))
    
    # plot 1
    ax[0].plot(df.index, df['Close'], label = 'Close', color = 'blue')
    ax[0].plot(df.index, df['SMA20'], label = 'SMA20', color = 'orange')
    ax[0].plot(df.index, df['SMA50'], label = 'SMA50', color = 'red')
    ax[0].plot(df.index, df['BB_upper'], label = 'BB-Upper', color = 'green', linestyle = '--')
    ax[0].plot(df.index, df['BB_lower'], label = 'BB-Lower', color = 'green', linestyle = '--')
    ax[0].legend()
    ax[0].set_title('Price + Moving Averages + Bollinger Bands')

    # plot 2
    ax[1].plot(df.index, df['RSI_14'], label = 'RSI_14')
    ax[1].axhline(70, color='red', linestyle='--')
    ax[1].axhline(30, color='green', linestyle='--')
    ax[1].set_title('RSI 14')


    #plot 3
    ax[2].plot(df.index, df['Volatility_20'])
    ax[2].set_title('Volatility_20')

    plt.tight_layout()
    plt.savefig('data/features_plot.png')

