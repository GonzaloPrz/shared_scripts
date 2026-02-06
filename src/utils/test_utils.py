import pandas as pd

from utils import Model

def test_model(model_class,params,scaler,imputer, X_dev, y_dev, X_test,problem_type='clf',fill_na=None):
    
    """
    Tests a model on specified development and test datasets, using optional bootstrapping for 
    training and evaluation, and calculates metrics based on provided criteria. Supports both 
    classification and regression.

    Parameters
    ----------
    model_class : class
        The model class (e.g., sklearn model) to instantiate for training and evaluation.
    params : dict
        Parameters to initialize the model.
    scaler : object
        Scaler instance to preprocess the feature data.
    imputer : object
        Imputer instance to handle missing values.
    X_dev : pd.DataFrame or np.array
        Development data features for training the model.
    y_dev : pd.Series or np.array
        Target values for the development dataset.
    X_test : pd.DataFrame or np.array
        Test data features for evaluating the model.
    problem_type : str, optional
        Specifies 'clf' for classification or 'reg' for regression tasks (default is 'clf').

    Returns
    -------
    outputs : np.array
        Array of model outputs for each test sample.
    """

    if not isinstance(X_dev, pd.DataFrame):
        X_dev = pd.DataFrame(X_dev)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    model = Model(model_class(**params),scaler,imputer)
    model.train(X_dev, y_dev,fill_na=fill_na)

    outputs = model.eval(X_test, problem_type,fill_na=fill_na)

    return outputs
