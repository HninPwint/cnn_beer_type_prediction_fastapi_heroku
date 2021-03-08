

def convert_to_date(df, cols:list):
    ''' Convert the specified columns of the dataframe into datetime
    
    Parameters
    ----------
    df : pd.dataframe
        Input dataframe
    cols: list
        List of columns to be converted

    Returns
    -------
    pd.DataFrame
        Pandas  dataframe with converted columns
    '''
    import pandas as pd
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            
    return df



