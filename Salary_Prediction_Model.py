import numpy as np 
import pandas as pd 
#from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression

def data_loading_and_cleaning():
    """drive.mount("/content/drive/")
    df=pd.read_csv('/content/drive/My Drive/Dataset/Salary.csv')"""
    df=pd.read_csv("Salary.csv")
    df.drop(["Age","Gender"],axis=1,inplace=True)
    df.dropna(how="any",inplace=True)
    df.drop_duplicates(inplace=True)
    df["Job Title"]=df["Job Title"].str.lower()
    df["Education Level"]=df["Education Level"].str.lower()
    return df

def train_data_preprocessing(df):
    x=df.drop("Salary",axis=1)
    y=df["Salary"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    ord=OrdinalEncoder(categories=[["bachelor's", "master's", 'phd']])
    x_train["Education Level"]=ord.fit_transform(x_train[["Education Level"]])
    ohe=OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    x_train_encode=ohe.fit_transform(x_train[["Job Title"]])
    encode_df=pd.DataFrame(x_train_encode,columns=ohe.get_feature_names_out(["Job Title"]))
    encode_df.drop(["Job Title_accountant"],axis=1,inplace=True)
    x_train=pd.concat([x_train.drop("Job Title",axis=1).reset_index(drop=True),encode_df],axis=1)
    y_train = y_train.reset_index()
    y_train.drop("index",axis=1,inplace=True)
    return x_train,y_train,ord,ohe

def test_data_preprocessing(x,ord,ohe):
    x["Education Level"]=ord.transform(x[["Education Level"]])
    encode=ohe.transform(x[["Job Title"]])
    encode_df=pd.DataFrame(encode,columns=ohe.get_feature_names_out(["Job Title"]))
    encode_df.drop("Job Title_accountant",axis=1,inplace=True)
    x=pd.concat([x.drop("Job Title",axis=1).reset_index(drop=True),encode_df],axis=1)
    return x
class Salary:
    def __init__(self):
        self.df = data_loading_and_cleaning()
        self.x_train,self.y_train,self.ord,self.ohe=train_data_preprocessing(self.df)
        self.model=LinearRegression()
        self.model.fit(self.x_train,self.y_train)
    def predict(self,x_test):
        if isinstance(x_test,list):
            x_test=pd.DataFrame([x_test],columns=["Education Level","Job Title","Years of Experience"])
        x_test = test_data_preprocessing(x_test,self.ord,self.ohe)
        return self.model.predict(x_test)
