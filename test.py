import datawig

df = datawig.utils.generate_df_string( num_samples=200, 
                                       data_column_name='sentences', 
                                       label_column_name='label')

df_train, df_test = datawig.utils.random_split(df)

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=['sentences'], # column(s) containing information about the column we want to impute
    output_column='label', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)