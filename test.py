from datawig.simple_imputer import SimpleImputer
import pandas as pd
import numpy as np
import mxnet as mx


if __name__ == "__main__":
    # generate a dataset with missing values for debugging using pandas and numpy
    df = pd.DataFrame(np.random.randn(10, 3), columns=list('ABC'))
    df.loc[df.sample(frac=0.1).index, 'A'] = np.nan
    df.loc[df.sample(frac=0.1).index, 'B'] = np.nan
    df.loc[df.sample(frac=0.1).index, 'C'] = np.nan
    df.head()
    
    # initialize a SimpleImputer model
    imputer = SimpleImputer(
        input_columns=['B', 'C'],  # columns containing information about the column we want to impute
        output_column='A',  # the column we'd like to impute values for
        output_path='imputer_model'  # stores model data and metrics
    )
    
    # fit an imputer model on the train data using gpu. Set the context to gpu.
    context = mx.gpu(0)
    
    imputer.fit(train_df=df, num_epochs=1, ctx=context)
    
