from datawig.simple_imputer import SimpleImputer
import pandas as pd
import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))


def replace_edu(rowdata):
    name = rowdata['required_education']

    if name == 'Some High School Coursework':
        return 'High School or equivalent'
    elif name == 'Vocational - HS Diploma':
        return 'High School or equivalent'
    elif name == 'Vocational - Degree':
        return 'High School or equivalent'
    elif name == 'Vocational':
        return 'High School or equivalent'
    elif name == 'Some College Coursework Completed':
        return "Bachelor's Degree"
    elif name == 'Doctorate':
        return "Professional"
    elif str(name) == 'nan':
        return "Unspecified"
    else:
        return name


def preprocess():
    df = pd.read_csv('./data/fake_job_postings.csv')
    df['required_education'] = df.apply(replace_edu, axis=1)
    df['location'] = df['location'].apply(lambda x : str(x).split(',')[0])
    df = df.drop(['department','salary_range'], axis = 1)
    
    df.to_csv("./data/fake_job_postings-before_imputation.csv", index=False)
    
    df = pd.read_csv("./data/fake_job_postings-before_imputation.csv", index_col=0)
    
    # check null ratio
    print(df.isnull().sum() / len(df))

def impute_train_whole(df, target_col_nm, token, chained):
    input_columns = list(set(df.columns) - set(target_col_nm))

    imputer = SimpleImputer(input_columns=input_columns,
                                    output_column=target_col_nm,
                                    tokens=token,
                                    output_path=f'{BASEDIR}/imputer/{token}/{chained}/{target_col_nm}'
                                    )

    imputer.fit(train_df=df, num_epochs=100, batch_size=128)

    imputed = imputer.predict(df)

    return pd.DataFrame(imputed)


def run_impute(df, token, chained):
    """
    If chanined is True, then impute the data sequentially, from low missing columns to high missing columns.
    If chanined is False, then impute the data independently, for each column. The original is then replaced with each imputed column.
    
    Token is either 'chars' or 'words'. It decides the unit of tf-idf vectorization.
    """
    target_col_nms = df.columns[df.isnull().any()].value_counts().sort_values(ascending=True).index.tolist()
    impute_result = {}
    temp = df.copy(deep=True)
    chained_nm = 'chained' if chained else 'unchained'

    for target_col_nm in target_col_nms:
        if chained:
            temp = df.copy(deep=True)

        print(f'Imputing {target_col_nm}...')
        imputed = impute_train_whole(temp, target_col_nm, token, chained_nm)
        print(f'Imputed {target_col_nm}...')

        impute_result[target_col_nm] = imputed

    final_df = df.copy(deep=True)

    for k, v in impute_result.items():
        final_df[k] = v[k]

    path = f'../data/imputed/{token}/{chained_nm}'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    final_df.to_csv(f"{path}/fake_job_postings.csv'", index=False)

    return final_df

def main():
    df = pd.read_csv("./data/fake_job_postings-before_imputation.csv", index_col=0)
    
    for chained in [True, False]:
        
        for token in ['chars', 'words']:
            run_impute(df, token, chained)
        
if __name__ == '__main__':    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"    # to disbale warning 'CUDA: invalid device ordinal', add the gpu number to the environment variable prior
    
    main()