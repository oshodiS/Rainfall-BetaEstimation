import pandas as pd
import numpy as np
from itertools import chain
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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


def geographic_plot(df, parameter1, parameter2):
    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot parameter1
    scatter1 = ax[0].scatter(df['X'], df['Y'], c=df[parameter1], cmap='viridis', s=10, alpha=0.8)
    ax[0].set_xlabel('longitude')
    ax[0].set_ylabel('latitude')
    ax[0].set_title(f'{parameter1} distribution')

    # Add colorbar for parameter1
    cbar1 = plt.colorbar(scatter1, ax=ax[0])
    cbar1.set_label(parameter1)

    # Plot parameter2
    scatter2 = ax[1].scatter(df['X'], df['Y'], c=df[parameter2], cmap='viridis', s=10, alpha=0.8)
    ax[1].set_xlabel('longitude')
    ax[1].set_ylabel('latitude')
    ax[1].set_title(f'{parameter2} distribution')

    # Add colorbar for parameter2
    cbar2 = plt.colorbar(scatter2, ax=ax[1])
    cbar2.set_label(parameter2)

    plt.show()


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
    
    inputs = inputs.reset_index()

    if start is None:
        start = 0
    if end is None:
        end = len(y)

    if distribution == 'beta':
        lb, ub = stats.beta.interval(confidence, a=dist.concentration1, b=dist.concentration0)
    elif distribution == 'gumbel':
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
    if distribution == 'beta':
        mmAMS_scaler = MinMaxScaler(feature_range=(0.001, 0.99))
        ams = std_df[['AMS']]
        scaled_ams = mmAMS_scaler.fit_transform(ams)
        std_df['AMS'] = scaled_ams
    
    return std_df

def evaluation(model, X,y, model_name, split):
    ''' Print the evaluation of the model and return the metrics'''
    
    print(f'Evaluating the {model_name} on {split} set...')
    dist = model(X)
    mae, ks_statist = sample_metrics(dist, y, split + ' - Ground Truth', 'red')
    #print(split,':')
    #print(f'MAE: {mae:.2f}')
    #print(f'KS statistics: {ks_statist:.2f}')
   
    return  [split, mae, ks_statist]

def parameters_metrics(dist, true_parameters, distribution_name = 'beta', indexes = None, plot = True, calculate_metrics = True, remove_outliers = False):

    if distribution_name == 'gumbel':
        param1_name_true = 'loc'
        param1_name_pred = 'loc_pred'
        param1_pred = dist.loc.numpy().ravel()
    
        param2_name_true = 'scale'
        param2_name_pred = 'scale_pred'
        param2_pred = dist.scale.numpy().ravel()
    
    elif distribution_name == 'beta':
        param1_name_true = 'alpha'
        param1_name_pred = 'alpha_pred'
        param1_pred = dist.concentration1.numpy().ravel()
    
        param2_name_true = 'beta'
        param2_name_pred = 'beta_pred'
        param2_pred = dist.concentration0.numpy().ravel()
    
    param1_true = true_parameters[param1_name_true].to_numpy()
    param1_max = param1_pred.max()
    param1_min = param1_pred.min()

    param2_true = true_parameters[param2_name_true].to_numpy()
    param2_max = param2_pred.max()
    param2_min = param2_pred.min()

    if remove_outliers:
        # remove from parameters the outliers in the true values
        out_indexes_param1 = np.where(param1_true > 40) #change value  
        if len(out_indexes_param1[0]) > 0:
            parameters = true_parameters.drop(out_indexes_param1)         
        out_indexes_param2 = np.where(param2_true > 200) #change value
        if len(out_indexes_param2[0]) > 0:
            parameters = parameters.drop(out_indexes_param2)
    
    if plot:
        parameters = true_parameters.copy()
        parameters[param1_name_pred] = param1_pred
        parameters[param2_name_pred] = param2_pred

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        sns.scatterplot(x=param1_name_pred, y=param1_name_true, data=parameters, hue='duration[h]', ax=ax[0], marker='o')
        x = np.linspace(param1_min, param1_max, 100)
        ax[0].plot(x, x, color='black', linestyle='--')
        ax[0].set_xlabel(param1_name_pred)
        ax[0].set_ylabel(param1_name_true + '_true')
        ax[0].set_title(f'Scatter plot of the predicted {param1_name_true} values')
        
        #ax[1].scatter(scale_pred, scale_true, color='blue', label='scale', marker='o')
        sns.scatterplot(x=param2_name_pred, y=param2_name_true, data=parameters, hue='duration[h]', ax=ax[1], marker='o')
        x = np.linspace(param2_min, param2_max, 100)
        ax[1].plot(x, x, color='black', linestyle='--')
        ax[1].set_xlabel(param2_name_pred)
        ax[1].set_ylabel(param2_name_true + '_true')
        ax[1].set_title(f'Scatter plot of the predicted {param2_name_true} values')
        plt.show()

    if calculate_metrics:
        
        metrics_param1_durations = {}
        metrics_param2_durations = {}
        
        param1_pred = parameters[param1_name_pred]
        param1_true = parameters[param1_name_true]

        param2_pred = parameters[param2_name_pred]
        param2_true = parameters[param2_name_true]

        param1_biasr_global = ((param1_true - param1_pred) / param1_true).mean()
        param1_rmse_global = metrics.mean_squared_error(param1_true, param1_pred)
        param1_pcc_global = np.corrcoef(param1_true, param1_pred)[0, 1]

        param2_biasr_global = ((param2_true - param2_pred) / param2_true).mean()
        param2_rmse_global = metrics.mean_squared_error(param2_true, param2_pred)
        param2_pcc_global = np.corrcoef(param2_true, param2_pred)[0, 1]

        metrics_param1_durations['global'] = [param1_biasr_global, param1_rmse_global, param1_pcc_global]
        metrics_param2_durations['global'] = [param2_biasr_global, param2_rmse_global, param2_pcc_global]

        for d in [1, 3, 6, 12, 24]:
            ids = indexes[d]

            param1_pred = parameters[param1_name_pred][ids]
            param1_true = parameters[param1_name_true][ids]

            param2_pred = parameters[param2_name_pred][ids]
            param2_true = parameters[param2_name_true][ids]

            param1_biasr = ((param1_true - param1_pred) / param1_true).mean()
            param1_rmse = metrics.mean_squared_error(param1_true, param1_pred)
            param1_pcc = np.corrcoef(param1_true, param1_pred)[0, 1]

            param2_biasr = ((param2_true - param2_pred) / param2_true).mean()
            param2_rmse = metrics.mean_squared_error(param2_true, param2_pred)
            param2_pcc = np.corrcoef(param2_true, param2_pred)[0, 1]

            metrics_param1_durations[d] = [param1_biasr, param1_rmse, param1_pcc]
            metrics_param2_durations[d] = [param2_biasr, param2_rmse, param2_pcc]

        metrics_param1_durations = pd.DataFrame(metrics_param1_durations, index=['biasr', 'rmse', 'pcc'])
        metrics_param1_durations['global'] = [param1_biasr_global, param1_rmse_global, param1_pcc_global]       
        
        metrics_param2_durations = pd.DataFrame(metrics_param2_durations, index=['biasr', 'rmse', 'pcc'])
        metrics_param2_durations['global'] = [param2_biasr_global, param2_rmse_global, param2_pcc_global]

        return metrics_param1_durations, metrics_param2_durations

    return None
   
def compare_samples(dist, distribution_name, parameters, index = 30):
    '''Compare the true and predicted samples of the distribution'''
    
    if(distribution_name == 'gumbel'):
        param1_pred = dist.loc.numpy().ravel()[index]
        param2_pred = dist.scale.numpy().ravel()[index]
        param1_name = 'loc'
        param2_name = 'scale'
        param1_true = parameters[param1_name][index]
        param2_true = parameters[param2_name][index]
        samples_pred = stats.gumbel_r.rvs(loc=param1_pred, scale=param2_pred, size=10000)
        samples_true = stats.gumbel_r.rvs(loc=parameters['loc'][index], scale=parameters['scale'][index], size=10000)

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



def get_comparison_plot(df1, df2, title, dist_name, colors):
    '''
    Function to plot the comparison of the results between the models.
    '''
    fig, axes = plt.subplots(2, 1, figsize=(13, 7)) 

    if(dist_name == 'gumbel'):
        param1 = 'loc'
        param2 = 'scale'
    elif(dist_name == 'beta'):
        param1 = 'alpha'
        param2 = 'beta' 
    
    # Plot for the first dataframe
    colors = sns.color_palette(colors, 4)
    df1.plot(kind='bar', ax=axes[0], rot=0, color=colors)
    axes[0].set_title(title +' Comparison - ' + param1)
    axes[0].set_xlabel('Models')
    axes[0].set_ylabel(title)
    axes[0].legend(title='metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    if (df1 < 0).all().all():
        axes[0].invert_yaxis()  # Invert y-axis if all values are negative
    axes[0].axhline(0, color='grey', linestyle='--')  # Add line at zero value

    # Plot for the second dataframe
    df2.plot(kind='bar', ax=axes[1], rot=0, color=colors)
    axes[1].set_title(title +' Comparison - ' + param2)
    axes[1].set_xlabel('Models')
    axes[1].set_ylabel(title)
    axes[1].legend(title='metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    if (df2 < 0).all().all():
        axes[1].invert_yaxis()  # Invert y-axis if all values are negative
    axes[1].axhline(0, color='grey', linestyle='--')  # Add line at zero value
    plt.tight_layout()

def get_global_results(metrics_name,models_name,models_metrics):    
    df_result = pd.DataFrame()
    for i, name_metrics in enumerate(metrics_name):
        for j in range(len(models_name)):
            df_result[name_metrics + "_" + models_name[j]  ] = models_metrics[j].iloc[i,:]
    return df_result




