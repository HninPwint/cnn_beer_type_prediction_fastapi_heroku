#A module to read raw data from source for further analysis.
#Data are stored in the raw_data folder for the original dataset and processed_data for the cleaned dataset.

from enum import Enum
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from imblearn.over_sampling import SMOTE

class BeerData(Enum):
    '''List of members to identify the name of each dataset stored in the raw_data folder.
       One enumerator member represents one file in the folder.
    '''
    RAW = 0

class DataReader:
    def __init__(self, relative_path = "../"):
    

        self.filepath = {}
        #raw data
        self.filepath.update({BeerData.RAW : relative_path + "data/raw/beer_reviews.csv"})

   
    def read_data(self, source, relative_path = "../"):
        '''Read the CSV file and load it into a data frame.
        
           Argument:
               source (enum): the member name of the dataset that will be loaded, e.g., BeerData.RAW 
           
           Return:
               a data frame that store the dataset
        '''
        if (not isinstance(source,BeerData)):
            raise Exception("argument should be filled with BeerData "
                            "Try BeerData.RAW")
            
        #read the data and load them into a dataframe
        data = pd.read_csv(self.filepath[source])
        #for consistency purposes, change the case of the column names into a lower case and remove extra spaces
        data = data.rename(columns = lambda x: x.strip(" "))
        return(data)

    def split_data(self, data, relative_path = "../"):
        ''' Split the given dataset into two sets randlomly
            such as Train and Validation sets by 80:20 ratio and 
            save the split data into the project's /data/processed directory 
            for later reuse during the experiment.
    
            Argument:
            ----------
            data : a panda dataframe
                A dataframe to split
            relative_path: a relative path string
                A relative path string to save the splited data
    
            Returns
            -------
            X_train: A panda dataframe with all the independent variables without the target column. 
                    It is 80% of the original dataset which is meant for model training
            X_val : A panda dataframe with all the independent variables. 
                    It is 20% of the original dataset which is meant for model validation
            y_train: A separate array of target column values from training set(X_train)
            y_val:  A separate array of target column values from validation set(X_val)      
        '''
        data = pd.DataFrame(data)
        target = data.pop('TARGET_5Yrs')
        X_train, X_val, y_train, y_val = train_test_split(data, target, test_size = 0.2, random_state=8, shuffle=True )

        # Save the splited data
        np.save(relative_path+ "data/processed/X_train", X_train)
        np.save(relative_path+ "data/processed/X_train", X_val)

        np.save(relative_path+ "data/processed/X_train", y_train)
        np.save(relative_path+ "data/processed/X_train", y_val)
        return(X_train, X_val, y_train, y_val)

    def select_feature_by_correlation(self, data, columns_to_drop):
        '''This function generates the correlation heat map
            select the features according to the correlation result.
            The features which have correlation > 0.9 are filter out.

            Argument:
            ----------
            data: a panda dataframe
                to examine the correlation among the features
            columns_to_drop: a list of columns 
                to exclude in the correlation analysis   

            Returns
            -------
            selected_columns: An array of selected columns which have correlation < 0.9 

        '''
        
        data.drop(columns_to_drop, axis=1, inplace=True )
        corr = data.corr()
        sns.heatmap(corr)

        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.9:
                    if columns[j]:
                        columns[j] = False
        selected_columns = data.columns[columns]
        return selected_columns

    
    def scale_features_by_standard_scaler(self, df):
        '''
            This function scales all the features included in the dataframe using Standard Scaler

            Arguments:
            ----------
            df: a panda dataframe with all the features to be scaled

            Return:
            -------
            scaled_df: a panda dataframe with all the scaled features keeping the original column names
        '''
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        
        # Set as pd.dataframe and Re-apply column names and
        scaled_df = pd.DataFrame(df_scaled)
        scaled_df.columns = df.columns
        return scaled_df
    
    def polynomialize_data(self, df, degree):

        '''
            This function transfrom and Generate a new feature matrix consisting of all polynomial combinations of the features with 
            degree less than or equal to the specified degree given in the second argument

            Arguments:
            ----------
            df: a panda dataframe with all the features to be polynomialised
            degree: integer
                    It is a degree to be applyed polynomial such as power 2 or 3

            Returns:
            -------
            data_poly: a panda dataframe
                 a dataframe with the polynomial combinations of the features     
        '''
        # Polynomialise
        poly = PolynomialFeatures(degree)
        data_poly = poly.fit_transform(df)
        
        data_poly = pd.DataFrame(data_poly)
        return data_poly
        
    
    def plot_class_balance(self, df):
        '''
            This function helps to visualises how balance the target class is 
            by plotting the bar graph on the target's value count

            Argument:
            --------
            df: a panda dataframe with the target feature

            Return:
            -------
            The bar plot
        '''
        
        pl = pd.DataFrame(df)
        pl.columns = ['Target']
        pl.Target.value_counts().plot(kind="bar", title="Count Target")
        
    def resample_data_upsample_smote(self, X, y):
        '''
            This function does oversampling the minority class using SMOTE from imblearn and 
            then fit and apply it in one step.

            Arguments:
            ----------
            X: a panda dataframe with the features only
            y: an array of target variable

            Returns:
            -------
            X_res:  a panda dataframe
                    a transformed version of the dataset with all features after upsampling
            y_res: an array of target values after upsampling

        '''
        
        sm = SMOTE(random_state = 23, sampling_strategy = 1.0)
        X_res, y_res = sm.fit_resample(X, y.ravel())
        return( X_res, y_res)
    
    def clean_negatives(self, strategy, df):
        '''
        This function does imputation to the dataset accorting to the strategy given in the arguemnt

        Arguments:
        ---------
        strategy: a string
                It is a strategy to apply to the imputation of the null / negative data.
                The strategy must be 'abs' to make the absolute value, 'null' to replace with 0 or
                'mean' to replace the negative with mean values.
        Returns:
        --------
        df: a panda dataframe
            A dataframe with features imputed
          
        '''
    
        if strategy=='abs':
            df = abs(df)
        if strategy=='null':
            df[df < 0] = None
        if strategy=='mean':
            df[df < 0] = None
            df.fillna(df.mean(), inplace=True)     
        if strategy=='median':
            df[df < 0] = None
            df.fillna(df.median(), inplace=True) 

        return(df)

# end of DataReader


def confusion_matrix(true, pred):
    '''
        This function plots confusion matrix
        
        Arguments:
        ---------
        true: an array of target's original value
        pred: an array of predicted values on target

        Result:
        -------
        cmtx: ndarray of shape (n_classes, n_classes) in a dataframe format
    '''

    import numpy as np
    import pandas as pd
    from sklearn import metrics

    unique_label = np.unique([true, pred])
    cmtx = pd.DataFrame(
        metrics.confusion_matrix(true, pred, labels=unique_label), 
        index=['true:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    return(cmtx)

def plot_roc(true, pred):
    '''
        This function plots confusion matrix
        
        Arguments:
        ---------
        true: an array of target's original value
        pred: an array of predicted values on target

        Result:
        -------
        cmtx: ndarray of shape (n_classes, n_classes) in a dataframe format
    '''

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, thresh = roc_curve(true, pred)
    roc_auc = auc(fpr, tpr)
    print('AUC = %0.3f' % roc_auc)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1,1.0])
    plt.xlim([-0.1,1.01])
    return(plt)

def eval_report(true, pred):
    '''
        This function generates confusion matrix, classification report and ROC curve
        
        Arguments:
        ---------
        true: an array of target's original value
        pred: an array of predicted values on target

        Prints:
        -------
        matrix: ndarray of shape (n_classes, n_classes) 
        matrix: Classification report in string format
        plot: ROC curve displaying the accuracy of the model
    '''
    import numpy as np
    import pandas as pd
    from sklearn import metrics

    unique_label = np.unique([true, pred])
    cmtx = pd.DataFrame(
        metrics.confusion_matrix(true, pred, labels=unique_label), 
        index=['true:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
    print("Confusion Matrix:")
    print(cmtx)
    print("")
    print("Classification Report:")
    print(metrics.classification_report(true, pred))
    print("")
    print("ROC Curve:")

    import matplotlib.pyplot as plt  
    plot_roc(true, pred)
    plt.show()
