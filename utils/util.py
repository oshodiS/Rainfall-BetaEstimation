import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from fitter import Fitter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tensorflow.keras import callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from scipy import stats 


def load_data(colab):
    '''
    Load the dataset from the csv file
    '''
    if colab:
        data_folder = Path.cwd() / "drive" / "MyDrive" / "A3I" / "data"
        dataset_path = data_folder.joinpath("AMS_descritt_noSM_meltD_adim.csv")
        dataset_indexes_test = data_folder.joinpath("gumMap_statbench_Gumfit_NEW.csv")
    else:
        dataset_path = "data/AMS_descritt_noSM_meltD_adim.csv"
        dataset_indexes_test = "data/gumMap_statbench_Gumfit_NEW.csv"

    df = pd.read_csv(dataset_path, sep=',', encoding='utf-8')
    df.drop(df.columns[[0]], axis=1, inplace=True)

    df_indexes_test = pd.read_csv(dataset_indexes_test, sep=',', encoding='utf-8') #dataset used to extract the ids for test
    return df, df_indexes_test


def make_pca(df, n_components):
    '''
    Apply PCA to the dataset
    '''
    pca = PCA(n_components=n_components)
    pca.fit(df)
    return pd.DataFrame(pca.transform(df))


def train_nn_model(model, X, y, batch_size, loss, lr, epochs, verbose=0, patience=30, validation_data=None, **fit_params):
    '''
    Train the neuro probabilistic model
    '''
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss)
    
    # Build the early stop callback
    cb = []
    if validation_data is not None:
        cb += [callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, mode='auto', min_delta=1e-3)]
        
    # Train the model
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=cb, validation_data=validation_data, verbose=verbose, **fit_params)
  
    return history

def plot_training_history(history=None, figsize=None, print_final_scores=True):
    '''
    Plot the training history of the model
    '''
    
    plt.figure(figsize=figsize)
    
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    if len(history.history.keys()) > 0:
        plt.legend()
    
    plt.xlabel('epochs')
    plt.grid(linestyle=':')
    plt.tight_layout()
    plt.show()
    
    if print_final_scores:
        trl = history.history["loss"][-1]
        s = f'Final loss: {trl:.4f} (training)'
        if 'val_loss' in history.history:
            vll = history.history["val_loss"][-1]
            s += f', {vll:.4f} (validation)'
        print(s)

def sample_metrics(dist, y, label, color, plot = True):
    '''
    Compute the mean absolute error and the Kolmogorov-Smirnov statistics
    '''
    
    num_samples = 1
    y_pred = dist.sample(num_samples).numpy().ravel()
    mae = metrics.mean_absolute_error(y, y_pred)
    ks_statistics, _ = stats.ks_2samp(y, y_pred)

    if plot:
        plt.hist(y, bins='auto', alpha=0.7, label=label, density=True, color='green');
        #sns.kdeplot(y_pred, label='Estimated sample', fill=True)
        plt.hist(y_pred, bins='auto', alpha=0.5, label='Predicted', density=True, color=color);
        plt.legend()
        plt.show()

    return mae, ks_statistics


def plot_series(data, labels=None, predictions=None, figsize=None, filled_version=None, std=None, ci=None, title=None, ylim=None):
    plt.figure(figsize=figsize)
    plt.plot(data.index, data.values, zorder=0)
    
    if filled_version is not None:
        filled = filled_version.copy()
        filled[~data['value'].isnull()] = np.nan
        plt.scatter(filled.index, filled, marker='.', c='tab:orange', s=5);
    
    # Plot standard deviations
    if std is not None:
        lb = data.values.ravel() - std.values.ravel()
        ub = data.values.ravel() + std.values.ravel()
        plt.fill_between(data.index, lb, ub, alpha=0.3, label='+/- std')
    
    # Plot confidence intervals
    if ci is not None:
        lb = ci[0].ravel()
        ub = ci[1].ravel()
        plt.fill_between(data.index, lb, ub, alpha=0.3, label='C.I.')
    
    # Rotated x ticks
    plt.xticks(rotation=45)
    
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels], color='red', zorder=2, s=5)
    
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions], color='black', alpha=.4, zorder=3, s=5)
    
    # Force y limits
    if ylim is not None:
        plt.ylim(*ylim)
    
    plt.grid(linestyle=':')
    plt.title(title)
    plt.tight_layout()


def percentage_in_ci(inputs, y, dist, confidence, distribution = 'beta', start = None, end = None, plot = True, num_samples=1):
    '''Compute the percentage of true values in the confidence interval'''
    
    if start is None:
        start = 0
    if end is None:
        end = len(y)

    if distribution == 'beta':
        lb, ub = stats.beta.interval(confidence, a=dist.concentration1, b=dist.concentration0)
        mean_dist = dist.mean().numpy().ravel()
        y_pred = dist.sample(num_samples).numpy().ravel()

        if plot:
            plot_series(pd.Series(index=inputs[start:end].index, data=mean_dist[start:end]), ci=(lb[start:end], ub[start:end]), figsize=(12,6))
            plt.scatter(inputs[start:end].index, y_pred[start:end], marker='o', color='blue', label='Predicted');
            plt.scatter(inputs[start:end].index, y[start:end], marker='x', color='red', label='True');
            plt.legend()
            plt.show()
    
        count_true = 0
        for i in range(len(y)):
            if lb[i] <= y[i] <= ub[i]:
                count_true += 1
        
        true_guess = count_true/len(y)*100

        return true_guess
    
    inputs = inputs.reset_index()
    if distribution == 'gumbel':
        lb, ub = stats.gumbel_r.interval(confidence, loc=dist.loc, scale=dist.scale)
        mean_dist = dist.mean().numpy().ravel()
        y_pred = dist.sample(num_samples).numpy().ravel()

        if plot:
            plot_series(pd.Series(index=inputs[start:end].index, data=mean_dist[start:end]), ci=(lb[start:end], ub[start:end]), figsize=(12,6))
            plt.scatter(inputs[start:end].index, y_pred[start:end], marker='o', color='blue', label='Predicted');
            plt.scatter(inputs[start:end].index, y[start:end], marker='x', color='red', label='True');
            plt.legend()
            plt.show()
    
        count_true = 0
        for i in range(len(y)):
            if lb[i] <= y[i] <= ub[i]:
                count_true += 1
        
        true_guess = count_true/len(y)*100

        return true_guess
    


def standardize(df, distribution):
    '''Standardize the dataset depending on the distribution '''

    if(distribution == 'gumbel'):
        features_not_to_scale = ['ID', 'AMS', 'mean_IdD', 'loc', 'scale', 'duration[h]']
        #scaler = MinMaxScaler() #min max scaler to have values in range [0, 1]
        scaler = StandardScaler() #standard scaler to have values with mean 0 and std 1
    else:
        features_not_to_scale = ['ID', 'AMS', 'mean_IdD', 'duration[h]']
        #different scalers 
        #scaler = MinMaxScaler()
        #scaler = RobustScaler(with_centering=False)
        scaler = StandardScaler()
    
    features_to_scale = df.columns.drop(features_not_to_scale)
    order_columns = features_not_to_scale + list(features_to_scale)
    scaled_data = scaler.fit_transform(df[features_to_scale])
    non_scaled_data = df[features_not_to_scale]
    std_df = np.concatenate([non_scaled_data, scaled_data], axis=1)

    # convert to dataframe
    std_df = pd.DataFrame(std_df, columns=order_columns)

    # scale AMS
    mmAMS_scaler = MinMaxScaler(feature_range=(0.001, 0.99))
    ams = std_df[['AMS']]
    scaled_ams = mmAMS_scaler.fit_transform(ams)
    std_df['AMS'] = scaled_ams
    
    return std_df

def evaluation(model, X,y, title):
    ''' Print the evaluation of the model and return the metrics'''
    
    print('Evaluating the models on ',title,'set...')
    dist = model(X)
    mae, ks_statist = sample_metrics(dist, y, title + ' - Ground Truth', 'red')
    print(title,':')
    print(f'MAE: {mae:.2f}')
    print(f'KS statistics: {ks_statist:.2f}')
   
    return  [title, mae, ks_statist]
   
def compare_samples(dist, distribution_name, parameters, index = 30):
    '''Compare the true and predicted samples of the distribution'''
    
    if(distribution_name == 'gumbel'):
        param1_pred = dist.loc.numpy().ravel()[index]
        param2_pred = dist.scale.numpy().ravel()[index]
        param1_name = 'loc'
        param2_name = 'scale'

    if(distribution_name == 'beta'):
        param1_pred = dist.concentration1.numpy().ravel()[index]
        param2_pred = dist.concentration0.numpy().ravel()[index]
        param1_name = 'alpha'
        param2_name = 'beta'

    param1_true = parameters[param1_name][index]
    param2_true = parameters[param2_name][index]
    samples_pred = stats.beta.rvs(a=param1_pred, b=param2_pred, size=10000)
    samples_true = stats.beta.rvs(a=param1_true, b=param2_true, size=10000)

    print('True',param1_name,': ', param1_true, 'Predicted ',param1_name,':',param1_pred)
    print('True',param2_name,': ', param2_true, 'Predicted ',param2_name,':', param2_pred)
    plt.hist(samples_true, bins='auto', alpha=0.7, label='True', density=True, color='green');
    plt.hist(samples_pred, bins='auto', alpha=0.5, label='Predicted', density=True, color='blue');
    plt.legend()
    plt.show()
