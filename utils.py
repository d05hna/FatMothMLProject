import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

import xgbost as xgb
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.inspection import permutation_importance
import shap 

from tqdm import tqdm 


def run_svm_analysis(df, feature_columns, target_column, test_size=0.2, 
                    kernel='rbf', C=1.0):
    """
    Support Vector Machine classifier on features and returns model and evaluation metrics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        dataframe of features and target
    feature_columns : list
        List of Features (str) used to subset the df, should be column names 
    target_column : str
        Name of the binary target column (for pre (0) and post (1) feeding)
    test_size : float, optional (default=0.2)
        test set size (prop of total data)
    kernel : str, optional (default='rbf')
        Kernel type to be used in the algorithm. Options: 'rbf', 'linear', 'poly', 'sigmoid'
    C : float, optional (default=1.0)
        HyperParameter for how strict we calculate error (1 to 10 inclusive)

        
    Returns:
    --------
    dict containing:
        'model': fitted SVM model
        'scaler': fitted StandardScaler object
        'train_score': training accuracy
        'test_score': testing accuracy
        'cv_scores': cross-validation scores
        'classification_report': detailed classification metrics
        'confusion_matrix': confusion matrix
        'predictions': test set predictions
        'X_test': scaled test features
        'y_test': test target values
    """
    # Input validation
    if not all(col in df.columns for col in feature_columns + [target_column]):
        raise ValueError("One or more specified columns not found in dataframe")
        
    if not set(df[target_column].unique()).issubset({0, 1}):
        raise ValueError(f"Target column '{target_column}' must contain only 0s and 1s")
    
    # Extract features and target
    X = df[feature_columns].values.astype(float)
    y = Binarizer().fit_transform(df[target_column].values.reshape(-1,1)).ravel()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the SVM
    svm = SVC(kernel=kernel, C=C,probability=True)
    svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = svm.predict(X_train_scaled)
    test_pred = svm.predict(X_test_scaled)
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        SVC(kernel=kernel, C=C),
        X_train_scaled, y_train, cv=5
    )
    
    # Calculate metrics
    train_score = svm.score(X_train_scaled, y_train)
    test_score = svm.score(X_test_scaled, y_test)
    
    return {
        'model': svm,
        'scaler': scaler,
        'train_score': train_score,
        'test_score': test_score,
        'cv_scores': cv_scores,
        'classification_report': classification_report(y_test, test_pred),
        'confusion_matrix': confusion_matrix(y_test, test_pred),
        'predictions': test_pred,
        'X_test': X_test_scaled,
        'y_test': y_test
    }

def train_xgboost_classifier(df, target_col, feature_cols, test_size=0.2,printout = True,max_depth=4,learning_rate=0.1):
    """
    Train an XGBoost classifier that handles Nan values (Sparese Data).
    
    Parameters:
    df: pandas DataFrame containing the data
    target_col: string, name of the target column (trial in the case of fatmoths) Should be 0 and 1 only 
    feature_cols: list names of feature columns (str)
    test_size: float, proportion of data to use for testing 
    
    Returns:
    model: trained XGBoost model
    X_test: test features
    y_test: test labels
    feature_importance: DataFrame of feature importance scores
    """
    
    # Prepare features and target
    X = df[feature_cols].astype(float)
    y = Binarizer().fit_transform(df[target_col].values.reshape(-1,1)).ravel()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
    )
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'evals': ['logloss', 'auc'],
        'silent': 1,
    }
    
    # Set up evaluation list
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    
    # Train the model
    num_rounds = 100
    bst = xgb.train(
        params,
        dtrain,
        num_rounds,
        evallist,
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    # Convert to sklearn-style classifier for convenience
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        missing=np.nan,
        max_depth=4,
        learning_rate=0.1,
        n_estimators=bst.best_ntree_limit if hasattr(bst, 'best_ntree_limit') else num_rounds,
    )
    
    # Fit the sklearn model for compatibility
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    if printout:
        # Print performance metrics
        print("\nModel Performance:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
        
    
    return model, X_test, y_test, feature_importance,roc_auc_score(y_test,y_pred_proba)



def plot_svm_results(results, feature_names, figsize=(15, 10)):
    """
    Plotting SVM results including:
    1. Feature importance plot (for linear kernel)
    2. Confusion matrix heatmap
    3. Cross-validation scores
    4. ROC curve (optional)
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_svm_analysis
    feature_names : list
        List of feature names
    figsize : tuple
        Figure size for the plots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('SVM Analysis Results', fontsize=16, y=1.02)
    
    # 1. Feature Importances Linear Kernel Only 
    if hasattr(results['model'], 'coef_'):
        importances = np.abs(results['model'].coef_[0])
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        ax = axes[0,0]
        ax.barh(np.arange(len(feature_importance)), feature_importance['importance'])
        ax.set_yticks(np.arange(len(feature_importance)))
        ax.set_yticklabels(feature_importance['feature'])
        ax.set_title('Feature Importance (Absolute Coefficients)')
        ax.set_xlabel('Absolute Coefficient Value')
    else:
        axes[0,0].text(0.5, 0.5, 'Feature importance only for linear kernel',
                      ha='center', va='center')
        axes[0,0].set_title('Feature Importance')
    
    # 2. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
    axes[0,1].set_title('Confusion Matrix')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('Actual')
    
    # 3. Cross-validation scores
    cv_scores = results['cv_scores']
    axes[1,0].boxplot(cv_scores)
    axes[1,0].set_title('Cross-validation Score Distribution')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                      label=f'Mean = {np.mean(cv_scores):.3f}')
    axes[1,0].legend()
    
    # 4. Training vs Testing Performance
    performance = pd.DataFrame({
        'Metric': ['Training', 'Testing', 'CV Mean'],
        'Accuracy': [results['train_score'], 
                    results['test_score'],
                    np.mean(results['cv_scores'])]
    })
    
    sns.barplot(x='Metric', y='Accuracy', data=performance, ax=axes[1,1])
    axes[1,1].set_title('Model Performance Comparison')
    axes[1,1].set_ylim(0, 1)
    
    
    for i, v in enumerate(performance['Accuracy']):
        axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    return fig

def plot_permutation_importance(perm_importance, feature_names):
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        })
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Permutation Importance (decrease in model score)')
        plt.title('Feature Importance (Permutation Method)')
        
        # Add error bars
        plt.errorbar(importance_df['importance'], range(len(importance_df)),
                    xerr=importance_df['std'], fmt='none', color='black', capsize=5)
        
        plt.tight_layout()
        return importance_df

def plot_shap_values(model, X, feature_names):
        # Create explainer
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100),)
        
        # Calculate SHAP values for a subset of data points
        shap_values = explainer.shap_values(X[:100])
        
        # If binary classification, take the second class's SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(10, 6))

        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap
        })
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance (SHAP Values)')
        
        plt.tight_layout()
        return feature_importance
    
    
def analyze_rbf_feature_importance(model, X, y, feature_names, n_repeats=10):
    """
    Analyzes feature importance for RBF SVM using multiple methods.
    
    Parameters:
    -----------
    model : trained SVM model
        The fitted SVM model with RBF kernel
    X : 
        Feature matrix
    y : 
        Target values
    feature_names : list(str)
        List of feature names
    n_repeats : int
        Number of times to repeat permutation importance calculation

    
    Returns:
    --------
    dict containing permutation importances and SHAP values
    """
    
    
    # Calculate permutation importance
    print("Calculating permutation importance...")
    perm_importance = permutation_importance(model, X, y, 
                                          n_repeats=n_repeats)
    
    # Plot permutation importance
    print("\nPlotting permutation importance...")
    perm_importance_df = plot_permutation_importance(perm_importance, feature_names)
    
    # Calculate and plot SHAP values
    print("\nCalculating and plotting SHAP values...")
    shap_importance_df = plot_shap_values(model, X, feature_names)
    
    # Create summary plot comparing both methods
    plt.figure(figsize=(12, 6))
    
    # Normalize importances to [0,1] for comparison
    perm_norm = (perm_importance_df['importance'] - perm_importance_df['importance'].min()) / \
                (perm_importance_df['importance'].max() - perm_importance_df['importance'].min())
    shap_norm = (shap_importance_df['importance'] - shap_importance_df['importance'].min()) / \
                (shap_importance_df['importance'].max() - shap_importance_df['importance'].min())
    
    comparison_df = pd.DataFrame({
        'feature': feature_names,
        'Permutation': perm_norm,
        'SHAP': shap_norm
    }).melt(id_vars=['feature'], var_name='Method', value_name='Normalized Importance')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Normalized Importance', y='feature', hue='Method', data=comparison_df)
    plt.title('Feature Importance Comparison (Normalized)')
    plt.tight_layout()
    
    return {
        'permutation_importance': perm_importance_df,
        'shap_importance': shap_importance_df,
        'comparison': comparison_df
    }


def make_violin(features, ax, title):
    """
    Helper Function to create violin plot on given axis

    """
    if len(features) > 0:
        melted = pd.melt(results_df[features], 
                        var_name='Feature', 
                        value_name='Importance')
        feature_order = (melted.groupby('Feature')['Importance']
                        .median()
                        .sort_values(ascending=False)
                        .index)
        
        sns.violinplot(data=melted,
                        x='Feature',
                        y='Importance',
                        order=feature_order,
                        inner='box',
                        scale='width',
                        ax=ax)
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

def plot_feature_importance_split(results_df, importance_threshold=0.05, figsize=(20, 12)):
    """
    Create two rows of violin plots: High and Low importance Featues based on some Given Threshold
    
    Parameters:
    results_df: DataFrame containing importance scores
    importance_threshold: float, threshold to split high/low  importance
    figsize: tuple for figure size
    
    Returns:
    The figure 
    """
    # Get feature columns
    feature_cols = [col for col in results_df.columns 
                   if col not in ['iteration', 'accuracy']]
    
    # Calculate mean importance
    mean_importance = results_df[feature_cols].mean()
    
    # Split features into high and low importance
    high_importance = mean_importance[mean_importance >= importance_threshold].index.tolist()
    low_importance = mean_importance[mean_importance < importance_threshold].index.tolist()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    
    # Create plots
    make_violin(high_importance, ax1, 
               f'High Importance Features (>= {importance_threshold})')
    make_violin(low_importance, ax2, 
               f'Low Importance Features (< {importance_threshold})')
    
    plt.suptitle('Feature Importance Distributions Across Trials', 
                y=1.02, fontsize=14)
    

    
    # Add summary statistics
    stats_text = (f"Number of trials: {len(results_df)}\n"
                 f"High importance features: {len(high_importance)}\n"
                 f"Low importance features: {len(low_importance)}\n"
                 f"Average Accuracy of trials: {np.round(np.mean(results['accuracy']),decimals=4)}")
    
    plt.figtext(0.85, 0.98, stats_text, 
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    return(fig)

def calculate_treeshap(model, X, feature_names=None, plot_summary=True):
    """
    Calculate TreeSHAP values for an XGBoost model and return plots if you want .
    
    Parameters:
    model: trained XGBoost model
    X:  DataFrame or numpy array of feature values
    feature_names: list of feature names (optional if X is a DataFrame)
    plot_summary: boolean, whether to display SHAP summary plot (defaults true because we wanna see the results)
    
    Returns:
    shap_values: numpy array of SHAP values for each prediction
    shap_df:  DataFrame containing mean absolute SHAP values per feature
    feature_importance:  DataFrame with both SHAP and native feature importance


    The Try in here is probably not needed as are some of the checks I do but I was getting weird errors 
    they are a relic from debugging but the code works now so idrc 
    """
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    
    # Convert DataFrame to DMatrix for XGBoost compatibility
    dmatrix = xgb.DMatrix(X, feature_names=X.columns.tolist())
    
    # Initialize the SHAP explainer with the underlying XGBoost model
    if hasattr(model, 'get_booster'):
        explainer = shap.TreeExplainer(model.get_booster())
    else:
        explainer = shap.TreeExplainer(model)
    
    try:
        # Calculate SHAP values using the DMatrix
        shap_values = explainer.shap_values(dmatrix)
        
        # If model is binary classifier, handle the output appropriately
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get positive class SHAP values
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame with feature importance metrics
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': mean_abs_shap,
            'mean_abs_shap_normalized': mean_abs_shap / mean_abs_shap.sum()
        }).sort_values('mean_abs_shap', ascending=True)  # Changed to True for bottom-to-top ordering
        
        # Get native feature importance from the model
        if hasattr(model, 'feature_importances_'):
            native_importance = pd.DataFrame({
                'feature': X.columns,
                'xgboost_importance': model.feature_importances_,
                'xgboost_importance_normalized': model.feature_importances_ / model.feature_importances_.sum()
            })
        else:
            # If using booster directly, get feature importance differently
            importance_type = 'gain'
            native_importance = pd.DataFrame({
                'feature': X.columns,
                'xgboost_importance': [0] * len(X.columns),
                'xgboost_importance_normalized': [0] * len(X.columns)
            })
        
        # Combine SHAP and native feature importance
        feature_importance = pd.merge(shap_df, native_importance, on='feature')
        
        # Create visualizations 
        if plot_summary:
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(shap_df))
            
            plt.barh(y_pos, shap_df['mean_abs_shap'], 
                    height=0.8, 
                    color='#3182bd',  
                    )
            
            plt.yticks(y_pos, shap_df['feature'])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance (SHAP Values)')
            
            # Remove top and right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            
            # Add gridlines
            plt.grid(axis='x', linestyle='-', alpha=0.2)
            
            plt.tight_layout()
            plt.show()
            
            # Create bar plot comparing SHAP vs XGBoost importance
            plt.figure(figsize=(12, 6))
            top_features = feature_importance.tail(20)  
            
            x = np.arange(len(top_features))
            width = 0.35
            
            plt.bar(x - width/2, top_features['mean_abs_shap_normalized'], 
                    width, label='SHAP Importance')
            plt.bar(x + width/2, top_features['xgboost_importance_normalized'], 
                    width, label='XGBoost Importance')
            
            plt.xlabel('Features')
            plt.ylabel('Normalized Importance')
            plt.title('SHAP vs XGBoost Feature Importance')
            plt.xticks(x, top_features['feature'], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        return shap_values, shap_df, feature_importance
    
    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        print("\nDebug information:")
        print(f"X shape: {X.shape}")
        print(f"X columns: {X.columns.tolist()}")
        print(f"Model type: {type(model)}")
        raise

def analyze_feature_interactions(shap_values, X, feature_names=None, top_k=5):
    """
    Analyze feature interactions using Tree SHAP interaction values.
    
    Parameters:
    shap_values: SHAP values from calculate_treeshap
    X: feature data
    feature_names: list of feature names
    top_k: number of top interactions to return
    
    Returns:
    interaction_df: DataFrame containing top feature interactions
    """
    if feature_names is None:
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
    
    # Calculate pairwise feature interactions
    interactions = np.zeros((len(feature_names), len(feature_names)))
    
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            interaction = np.abs(shap_values[:, i] * shap_values[:, j]).mean()
            interactions[i, j] = interaction
            interactions[j, i] = interaction
    
    # Create DataFrame of interactions
    interaction_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            interaction_pairs.append({
                'feature1': feature_names[i],
                'feature2': feature_names[j],
                'interaction_strength': interactions[i, j]
            })
    
    interaction_df = pd.DataFrame(interaction_pairs)
    interaction_df = interaction_df.sort_values('interaction_strength', ascending=False).head(top_k)
    
    return interaction_df