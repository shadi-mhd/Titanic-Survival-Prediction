import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    for col in ['Sex', 'Embarked']:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col])
    
    return df
