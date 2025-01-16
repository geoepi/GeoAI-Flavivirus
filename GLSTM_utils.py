import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import Queen
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC
from imblearn.over_sampling import RandomOverSampler
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.colorbar import ColorbarBase
import seaborn as sns



def merge_data(earthenv, topographic, habitat_hetero, reduced_features, ndvi):
    """
    Merge all data into a single dataframe.

    Args:
    1. earthenv
    2. topographic
    3. habitat_hetero
    4. reduced_features
    5. ndvi

    Returns:
    1. A df containing all features aligned to counties, years, and weeks.
    """
    class_columns = [f'class_{i}' for i in range(1, 13)]
    earthenv[class_columns] = earthenv[class_columns].clip(1e-6, 1 - 1e-6)
    # Apply logit transform to each of the class columns
    earthenv[class_columns] = earthenv[class_columns].apply(lambda x: np.log(x / (1 - x)))
    habitat_hetero = habitat_hetero.rename(columns = {'fips' : 'FIPS'})
    ndvi = ndvi.rename(columns = {'fips' : 'FIPS'})
    earthenv = earthenv.rename(columns = {'fips' : 'FIPS'})
    topographic = topographic.rename(columns = {'fips':'FIPS'})
    merge1 = pd.merge(reduced_features, earthenv, how = 'left', on='FIPS')
    merge2 = pd.merge(merge1, habitat_hetero, how='left', on='FIPS')
    merge3 = pd.merge(merge2, topographic, how='left', on='FIPS')
    # Reshape the DataFrame using pd.melt()
    ndvi_melted = pd.melt(ndvi, id_vars=['FIPS'], var_name='Year', value_name='Value')
    # Clean up the 'Year' column to keep only the year part
    ndvi_melted['Year'] = ndvi_melted['Year'].str.extract('(\d{4})')
    ndvi_melted['Year'] = ndvi_melted['Year'].astype(int)
    all_merge = pd.merge(merge3, ndvi_melted, on=['FIPS', 'Year'])

    return all_merge


def align(df, gdf, year= None):
    """
    Initial preprocessing steps are taken here. 

    Args: 
    1. df : A dataframe (df) that contains the features
    2. gdf : A geodataframe for the study domain
    3. year = None processes all years, year = sample_year drops sample_year
    
    Returns:
    1. df : a dataframe that contains necessary columns for the desired years.
        
    """
    if year is not None:
        filtered_df = df[df['Year'] == year]
    else: 
        filtered_df = df.copy()
    
    gdf = gdf.rename(columns = {'fips':'FIPS'})
    gdf['node_id']=gdf.index.values
    merge = pd.merge(filtered_df, gdf, on = 'FIPS', how='right')
    merge['Binary'] = np.where(merge['EQU'] >= 1, 1, 0)
    df = merge.drop(columns = ['OBJECTID', 'NAME','STATE_NAME','geometry', 'Area.S','s_DEM','s.EstPop','s.Pov_pct','s.Med_income', 'EQU','Month'])
    df['Index'] = df.index
    df['Index'], df['Binary'] = df['Binary'], df['Index']
    df = df.rename(columns={'Index': 'Binary', 'Binary': 'Index'})
    
    return df

def plot_correlation_with_target(df, target_variable):
    """
    Plots the correlation of features with a specified target variable.

    Arguments:
    1. df : The DataFrame containing the data.
    2. target_variable : The name of the target variable for correlation analysis.
    """
    # Calculate the correlation of the target variable with all features
    correlation_with_target = df.corr()[target_variable].drop(target_variable)

    # Convert to DataFrame for visualization
    correlation_df = correlation_with_target.reset_index()
    correlation_df.columns = ['Feature', 'Correlation']

    # Sort by correlation values in descending order
    correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

    # Set up the matplotlib figure with adjusted height for spacing
    plt.figure(figsize=(12, 10))

    # Create a bar plot to visualize correlation with the target variable
    sns.barplot(x='Correlation', y='Feature', data=correlation_df)

    # Add titles and labels
    plt.title(f'Correlation with Target Variable: {target_variable}', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Rotate y-ticks for better readability
    plt.yticks(rotation=30)

    # Show the plot
    plt.axvline(0, color='gray', linewidth=0.8)  # Add a vertical line at 0 for reference
    plt.tight_layout()  # Adjust layout to make room for the rotated labels
    plt.show()



def get_neighbors(gdf):
    """
    Generates adjacency matrix.

    Args:
    1. gdf: A geodataframe for the study domain

    Retruns:
    1. adj_matrix: Adjacency matrix 
    """
    Q_neighbors = Queen.from_dataframe(gdf)
    neighbors = {idx : gdf.iloc[neigh].index.tolist() for idx, neigh in Q_neighbors.neighbors.items()}
    rows =[]
    for source, target in neighbors.items():
        for target in target:
            rows.append({'source': source, 'target' : target})
    neighbors_df = pd.DataFrame(rows)
    G = nx.from_pandas_edgelist(neighbors_df, source='source', target='target', create_using=nx.DiGraph)
    adj_matrix = nx.to_numpy_array(G)
    return adj_matrix



def resample_and_order(df, resample=True):

    """
    Resamples (in particular function is written to perform oversampling), and ensures proper temporal ordering which is necessary for recurrent layers.

    Args:
    1. df : Output of align 

    Returns:
    1. ordered : A dataframe that ensures correcting temporal and spatial ordering
    2. resampled_indices : Indexes after oversampling and ordering, to be used to track indices from the test dataset for mapping results
    """
    if resample:
        ros = RandomOverSampler(random_state=0) # resample needed for training and testing datasets, not needed for validation.
        X_resampled, y_resampled = ros.fit_resample(df.drop(columns=[ 'Binary']), df['Binary'])
        to_order = X_resampled.merge(y_resampled, left_index=True, right_index=True)
        ordered = to_order.sort_values(['Year', 'Week', 'node_id'])
        ordered = ordered.reset_index()
        ordered['Index'] = ordered.index.values
        to_match = ordered.copy()
        ordered = ordered.drop(columns = ['index','FIPS'])
        resampled_indices = ordered.loc[ros.sample_indices_, 'Index'].values
    else:
        ordered = df.sort_values(['Year', 'Week', 'node_id'])
        ordered = ordered.reset_index()
        ordered['Index'] = ordered.index.values
        ordered = ordered.drop(columns = ['index'])
        resampled_indices = ordered.index.values

    
    feature_names = ordered.iloc[:,:].columns.tolist()
    
    return ordered, resampled_indices, to_match, feature_names


def split_normalize_format(ordered, resampled_indices, adj_matrix, split=True):
    """
    Peforms train/test split, normalization, and creates pytorch geometric data objects required by GLSTM model variants.

    Args:
    1. ordered : output of resample_and_order
    2. resampled_indices : output of resample_and_order
    3. adj_matrix : output of get_neighbors
    4. split = True/False : True -> split, False - > don't split.

    Returns:
    If Split = True:
    1. data_train : Training data. Features = data_train.x, labels = data_train.y, edge indices = data_train.edge_index
    2. data_test : Testing data. Features = data_test.x, labels = data_test.y, edge indices = data_test.edge_index
    3. X_train_normalized : Used for input shape
    4. node_id_test : Indices of test data, used for mapping results later.
    """
    if split:
        X_train, X_test, y_train, y_test, node_id_train, node_id_test = train_test_split(ordered.iloc[:,0:-1], ordered.iloc[:,-1], resampled_indices, test_size=0.2, random_state=42, shuffle = False)
        y_train = y_train.values
        y_test = y_test.values
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
    

        # Convert NumPy array to torch tensor
        adj_matrix_tensor = torch.tensor(adj_matrix)

        # Find the indices of non-zero elements in the adjacency matrix
        edge_indices = np.transpose(np.nonzero(adj_matrix))
    
        # Convert NumPy array to torch tensor
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Create PyTorch Geometric Data object
        data_train = Data(x=torch.tensor(X_train_normalized, dtype=torch.float),
            y=torch.tensor(y_train, dtype=torch.long),
            edge_index=edge_index_tensor,
            num_nodes=X_train_normalized.shape[0],
            num_classes=2)

        # Create PyTorch Geometric Data object for testing data
        data_test = Data(x=torch.tensor(X_test_normalized, dtype=torch.float),
            y=torch.tensor(y_test, dtype=torch.long),
            edge_index=edge_index_tensor,
            num_nodes=X_test_normalized.shape[0],
            num_classes=2) 
        node_id_test = X_test.index.values
        return data_train, data_test, X_train_normalized, node_id_test
    else: 
        X_val = ordered.iloc[:,0:-1]
        y_val = ordered.iloc[:,-1]
        y_val = y_val.values
        scaler = StandardScaler()
        X_val_normalized = scaler.fit_transform(X_val)

        # Convert NumPy array to torch tensor
        adj_matrix_tensor = torch.tensor(adj_matrix)

        # Find the indices of non-zero elements in the adjacency matrix
        edge_indices = np.transpose(np.nonzero(adj_matrix))
    
        # Convert NumPy array to torch tensor
        edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        
        
        # Create PyTorch Geometric Data object
        data_val = Data(x=torch.tensor(X_val_normalized, dtype=torch.float),
            y=torch.tensor(y_val, dtype=torch.long),
            edge_index=edge_index_tensor,
            num_nodes=X_val_normalized.shape[0],
            num_classes=2)

        node_id_val = X_val.index.values

        return data_val, X_val_normalized, node_id_val


def calculate_feature_importance(model, data, node_idx, feature_names, device='cpu'):
    """
    Calculates feature importance by iterating over evaluating the model on subsets of features.
    
    Arguments: 
    1. model: instantiated model
    2. data: collection of features to determine importance of
    3. node_idx: node index of those features
    4. feature_names: list of feature names
    5. device: cpu or gpu
    Returns:
    1. importance_dict: an importance dictionary mapping feature names to their importances.
    """
    model.eval()
    node_features = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    # Get predictions for original features
    original_predictions = model(node_features, edge_index).detach()
    
    # Calculate feature importances
    feature_importances = []
    for i in range(node_features.size(1)):  # Iterate over features
        perturbed_features = node_features.clone()
        perturbed_features[:, i] = 0  # Zero out one feature at a time
        perturbed_predictions = model(perturbed_features, edge_index).detach()
        
        # Measure change in prediction (e.g., using a loss or difference metric)
        importance = torch.abs(original_predictions - perturbed_predictions).mean().item()
        feature_importances.append(importance)
    
    # Map feature importance to feature names
    importance_dict = dict(zip(feature_names, feature_importances))
    return importance_dict


def plot_feature_importance(importance_dict, threshold, name_mapping):
    """
    Plots feature importance on a horizontal bar chart, highlighting features exhibiting variance above the threshold in green.

    Arguments:
    1. importance_dict: output from calculate_feature_importance
    2. threshold
    3. name_mapping: maps feature names in dataset to descriptive names
    """
    
    # Sort the importance dictionary #updated for renaming
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_importance]
    importances = [item[1] for item in sorted_importance]
    
    # Apply name mapping if provided
    if name_mapping:
        display_features = [name_mapping.get(feature, feature) for feature in features]
    else:
        display_features = features
    
    # Normalize to percentages
    total_importance = sum(importances)
    importances_percent = [(imp / total_importance) * 100 for imp in importances]
    
    # Assign colors based on threshold
    colors = ['green' if importance > threshold else 'blue' for importance in importances_percent]
    
    # Create the bar chart
    plt.figure(figsize=(12, 14))
    plt.barh(display_features, importances_percent, color=colors)
    plt.xlabel('Importance (%)',fontsize=14)  # Updated label to indicate percentages
    plt.title(f'Feature Importance (Threshold: {threshold}%)',fontsize=16)
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.gca().tick_params(axis='y', labelsize=12, pad=20)  # Adjust y-axis label font size
    plt.xticks(fontsize=12)  # Adjust x-axis ticks font size
    plt.yticks(fontsize=12)  
    
    # Adjust y-axis label padding
    plt.gca().tick_params(axis='y', pad=10)
    plt.show()


def plot_gradient_importance(param_names, gradients, name_mapping=None):
    """
    Function to plot the gradient flow and order from greatest to least importance.
    
    Args:
    param_names (list): List of parameter names.
    gradients (list): List of gradient values corresponding to the parameters.
    name_mapping (dict, optional): A dictionary to map parameter names to more descriptive labels.

    """
    
    # Optionally map parameter names using name_mapping
    if name_mapping:
        param_names = [name_mapping.get(param, param) for param in param_names]
    
    # Sort gradients and parameter names based on gradient values (descending)
    sorted_gradients, sorted_param_names = zip(*sorted(zip(gradients, param_names), reverse=False))
    
    # Plotting gradients
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_param_names, sorted_gradients, color='blue')
    plt.xlabel('Gradient Norm',fontsize=14)
    plt.ylabel('Parameter', fontsize=14)
    plt.title('Gradient Norms', fontsize=16)
    plt.tight_layout()
    plt.show()




def plot_loss_accuracy(train_loss_history, train_acc_history, val_loss_history=None, val_acc_history=None):

    """
    Function that plots training and testing loss and accuracy curves.

    Args: All arguments are outputs saved from the training loop.
    1. train_loss_history
    2. train_acc_history
    3. val_loss_history
    4. val_acc_history

    """

    epochs = range(1, len(train_loss_history) + 1)
    
    # Plotting Training Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label='Training Loss', marker='o')
    if val_loss_history:
        plt.plot(epochs, val_loss_history, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plotting Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, label='Training Accuracy', marker='o')
    if val_acc_history:
        plt.plot(epochs, val_acc_history, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('GLSMT4final_loss_acc.png')
    plt.tight_layout()
    plt.show()


def get_predictions(model, data):
    """
    Evaluates model on a set of data.

    Args:
    1. model : model used to train
    2. data : data to evaluate on
    
    Returns: 
    1. pred
    2. data.y
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        #predictions_with_ids = list(zip(node_id_test, pred.tolist()))
    return pred, data.y



def conf_mat(y_pred, y_true):
    """
    Function to display confusion matrix.

    Args:
    1. y_true: True class labels
    2. y_pred: Predicted class labels

    """

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


def plot_roc_curve(y_true, y_pred):
    """
    Function that plots ROC curve and computes and displays AUROC score.

    Args:
    1. y_true : output from confusion_matrix; ground truth class labels
    2. y_pred : output from confusion_matrix: predicted class labels
    
    """
    # Compute AUROC
    auroc = BinaryAUROC()
    roc_score = auroc(torch.tensor(y_pred), torch.tensor(y_true))
    print(f"ROC AUC Score: {roc_score.item()}")

    # Compute FPR and TPR
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_score.item():.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def match_predictions(model, data, node_id_test, df, gdf, years):
    """
    Function to match predictions (model output) with GDF needed for plotting.

    Args : 
    1. model
    2. data : data_test
    3. node_id_test : output from split_normalize_format
    4. df : use ordered (output from resample_and_order)
    5. gdf : study domain shapefile
    6. years: Single year (int) or list of years (list of int)

    Returns : 
    year_filtered : A GDF with a new column for predicted class labels for the specified year(s).
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        predictions_with_ids = list(zip(node_id_test, pred.tolist()))

    predictions_with_idsDF = pd.DataFrame(predictions_with_ids)
    predictions_with_idsDF=predictions_with_idsDF.rename(columns = {0:'Index', 1: 'ClassLabel'})
    merge = pd.merge(df, predictions_with_idsDF, on = 'Index', how ='left')
    merge2 = gdf.merge(merge, on = 'FIPS')

    # Handle single year and multiple years input
    if isinstance(years, int):
        years = [years]  # Convert single year to a list

    year_filtered = merge2[merge2['Year'].isin(years)]
    year_filtered['Difference'] = year_filtered['Binary'].astype(int) - year_filtered['ClassLabel'].astype(int)
        
    
    return year_filtered



def keep_most_frequent(df, group_cols, freq_col):
    # Create a temporary column with the frequency of each value in 'freq_col' within each group
    df['freq'] = df.groupby(group_cols)[freq_col].transform(lambda x: x.value_counts().max())

    # Sort by the frequency column in descending order
    df = df.sort_values(by=freq_col, ascending=False)

    # Drop duplicates based on group_cols, keeping the first occurrence (the one with the highest frequency)
    df_unique = df.drop_duplicates(subset=group_cols, keep='first')

    return df_unique


def visualize_filtered_graph(adj_matrix, degree_threshold=5, node_size=100, alpha=0.6):
    """
    Visualizes a filtered force-directed graph based on an adjacency matrix.

    Parameters:
    - adj_matrix: np.ndarray or list-like
        The adjacency matrix representing the graph.
    - degree_threshold: int
        The minimum degree for nodes to be included in the filtered graph.
    - node_size: int
        The size of the nodes in the graph.
    - alpha: float
        The transparency level of the nodes.
    """
    # Create the graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # Filter to keep only high-degree nodes
    filtered_nodes = [node for node, degree in dict(G.degree()).items() if degree > degree_threshold]
    H = G.subgraph(filtered_nodes)  # Create a subgraph

    # Draw the filtered graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(H)  # Use spring layout for better distribution

    # Draw nodes with transparency
    nx.draw_networkx_nodes(H, pos, node_size=node_size, node_color='lightblue', alpha=alpha)

    # Draw edges
    nx.draw_networkx_edges(H, pos, alpha=0.5)

    # Draw labels for the filtered nodes
    labels = {node: node for node in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

    plt.title('Filtered Force-Directed Graph Visualization')
    plt.axis('off')
    plt.show()

def prepare_for_mapping(df, counties_gdf, states_gdf ):
    df_unique['Difference'] = df_unique['Binary'].astype(int) - df_unique['ClassLabel'].astype(int)
    results_agg = df_unique.groupby('node_id').agg({ 'Binary': sum, 'ClassLabel':sum, 'Difference':sum, 'FIPS': 'first'}).reset_index()
    results_agg_gdf = gpd.GeoDataFrame.merge(south_counties, results_agg, on='FIPS', how='left')
    # If the CRSs are different, reproject the overlay to match the base GeoDataFrame's CRS
    if df_unique.crs != states_gdf.crs:
        print("CRS mismatch detected! Reprojecting south_states to match base GeoDataFrame CRS...")
        states_gdf = states_gdf.to_crs(results_agg_gdf.crs)

    return results_agg_gdf, states_gdf

def map_results(year, gdf, df, south_states):
    """
    A wrapper function to call plot_data and map the predicted model results.

    Args : 
    1. year : desired year to visualize results of
    2. gdf : GDF of study domain
    3. model :
    4. data :

    Returns:
    1. Displays two plots, one for each season (weeks 14-26, and 27-40)
    
    """
    #preds = match_predictions(model, data)
    #preds_group1, preds_group2 = results_per_year(preds, year, gdf)  
    
    # Define the bin edges and labels
    bin_edges = [-10,0, 1, 2, 3, 4, 5, np.inf]
    bin_labels = ['-1','0', '1', '2', '3', '4', '5+']

    # Function to plot data
    def plot_data(df, title_suffix, save_suffix, south_states):
        # Bin the data
        df['binned_labels'] = pd.cut(df['ClassLabel'], bins=bin_edges, labels=bin_labels, right=False)
        df['binned_truth'] = pd.cut(df['Binary'], bins=bin_edges, labels=bin_labels, right=False)
        df['binned_diffs'] = pd.cut(df['Difference'], bins=bin_edges, labels=bin_labels, right=False)

        discrete_colors = ['blue','lightgray', 'yellow', 'gold', 'darkorange', 'red', 'darkred']
        discrete_cmap = ListedColormap(discrete_colors)

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 3, figsize=(14, 7))

        # First subplot
        df.plot(column='binned_labels', cmap=discrete_cmap, linewidth=0.8, ax=axs[0], edgecolor='0.6')
        south_states.plot(ax=axs[0], linewidth=1.5, facecolor = 'none', edgecolor='0.5')  # Add state boundaries
        axs[0].set_title('a. Predictions')
        axs[0].axis('off')

        # Second subplot
        df.plot(column='binned_truth', cmap=discrete_cmap, linewidth=0.8, ax=axs[1], edgecolor='0.6')
        south_states.plot(ax=axs[1], linewidth=1.5, facecolor='none',edgecolor='0.5')  # Add state boundaries
        axs[1].set_title('b. Reported')
        axs[1].axis('off')

        # Third subplot
        df.plot(column='binned_diffs', cmap=discrete_cmap, linewidth=0.8, ax=axs[2], edgecolor='0.6')
        south_states.plot(ax=axs[2], linewidth=1.5, facecolor = 'none', edgecolor='0.5')  # Add state boundaries
        axs[2].set_title('c. Differences')
        axs[2].axis('off')

        # Add colorbar
        cax = fig.add_axes([0.00005, 0.05, 0.02, 0.8])
        norm = Normalize(vmin=-1, vmax=6)
        cb = ColorbarBase(cax, cmap=discrete_cmap, norm = norm )
        cb.set_ticks([-1, 0, 1, 2, 3, 4, 5])
        cb.set_ticklabels(['-1','0', '1', '2', '3', '4', '5 +'])
        #cb.set_label('Cumulative Weeks with Disease Presence', fontsize=10)

        # Adjust layout
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.001)

        # Adjust layout to prevent overlap
        fig.suptitle(f'{year}')
        plt.tight_layout()

        # Save the plot
        #plt.savefig(f'GLSTM42_{year}_map{save_suffix}.png')
        plt.show()

    # Plot for preds_group1
    plot_data(df, 'Group 1', '1', south_states)

    # Plot for preds_group2
    #plot_data(preds_group2, 'Group 2', '2')

def map_truth(gdf, preds_group1, preds_group2, col):
    """
    A function to plot ground truth of a column. 

    Args : 
    1. gdf : GDF of study domain
    2. preds_group1 : not actually predictions, just aggregated ground truth weeks 14-26
    3. preds_group2 : ditto but 27-40\
    4. col :  should match col from filter_truth

    Returns:
    1. Displays two plots, one for each season (weeks 14-26, and 27-40)
    
    """
    #preds = match_predictions(model, data)
    #preds_group1, preds_group2 = results_per_year(preds, year, gdf)  
    
    # Define the bin edges and labels
    bin_edges = [0, 1, 4, 11, 21, 51, np.inf]
    bin_labels = ['0', '1-3', '4-10', '11-20', '21-49', '51+']

        # Bin the data
    preds_group1['binned_truth'] = pd.cut(preds_group1[col], bins=bin_edges, labels=bin_labels, right=False)
    preds_group2['binned_truth'] = pd.cut(preds_group2[col], bins=bin_edges, labels=bin_labels, right=False)

    discrete_colors = ['lightgray', 'yellow', 'gold', 'darkorange', 'red', 'darkred', 'purple']
    discrete_cmap = ListedColormap(discrete_colors)

        # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # First subplot
    preds_group1.plot(column='binned_truth', cmap=discrete_cmap, linewidth=0.8, ax=axs[0], edgecolor='0.6')
    #counties.plot(ax=axs[0], edgecolor='lightgrey', color='0.9', alpha=0.5)
    axs[0].set_title('April - June')
    axs[0].axis('off')

        # Second subplot
    preds_group2.plot(column='binned_truth', cmap=discrete_cmap, linewidth=0.8, ax=axs[1], edgecolor='0.6')
    #counties.plot(ax=axs[1], edgecolor='lightgrey', color='0.9', alpha=0.5)
    axs[1].set_title('July - September')
    axs[1].axis('off')

        # Add colorbar
    cax = fig.add_axes([0.00005, 0.05, 0.02, 0.8])
    cb = ColorbarBase(cax, cmap=discrete_cmap)
    cb.set_ticklabels(['0', '1-3', '4-10', '11-20', '21-50', ' 51+'])
    #cb.set_label('Cumulative Disease Counts', fontsize=10)

        # Adjust layout
    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.001)

        # Adjust layout to prevent overlap
        #fig.suptitle(f'West Nile Virus ')
    plt.tight_layout()

        # Save the plot
    plt.savefig(f'cummulative.png')
    plt.show()

def filter_truth(df, gdf, col):
    """
    A function to filter weeks into seasons and aggregate a desired columns values. Used as input for map_truth.

    Args:
    1. df
    2. gdf
    3. col

    Returns: Two gdfs sorted by seasons with aggregated columns. 
    """
    filtered_results1 = results_year[(df['Week'] >= 14) & (df['Week'] <= 26)]
    filtered_results2 = results_year[(df['Week'] >= 27) & (df['Week'] <= 40)]
    #aggregate weeks 14-26, 27-40
    results_agg1 = filtered_results1.groupby(['FIPS']).agg({ col: sum}).reset_index()
    #aggregate weeks 14-26, 27-40
    results_agg2 = filtered_results2.groupby(['FIPS']).agg({ col: sum}).reset_index()
    results_agg1_gdf = gdf.merge(results_agg1, on='FIPS', how='right')
    results_agg2_gdf = gdf.merge(results_agg2, on='FIPS', how='right')
    return results_agg1_gdf, results_agg2_gdf
