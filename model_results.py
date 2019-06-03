import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def get_results(pred_df, actual_df):
    all_predictions = []
    all_results = []
    for col in list(pred_df.columns):
        all_predictions.append(pred_df[col].values)
        all_results.append(actual_df[col].values)
    all_predictions = np.concatenate(all_predictions)
    all_results = np.concatenate(all_results)

    print("Mean Squared Error")
    errors = all_predictions - all_results
    print(np.mean(np.square(errors)))

    print("Line")
    model = LinearRegression().fit(np.reshape(all_results,(-1,1)), all_predictions)
    print("Slope: "+str(model.coef_[0]))
    print("Intercept: "+str(model.intercept_))

    plt.plot(all_results, all_predictions, 'k.')
    plt.title("Correlation: " + str(np.correlate(all_results, all_predictions)[0]))
    plt.show()

    plt.hist(errors)
    plt.title("Histogram of Error")
    plt.show()
