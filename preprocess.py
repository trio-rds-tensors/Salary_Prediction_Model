class preprocessor (BaseEstimator,TransformerMixin):
    def __init__(self):
        self.ord=OrdinalEncoder(categories=[["bachelor's", "master's", 'phd']])
        self.ohe=OneHotEncoder(handle_unknown="ignore",sparse_output=False)

    def fit(self,x_train,y_train=None):
        self.ord.fit(x_train[["Education Level"]])
        self.ohe.fit(x_train[["Job Title"]])
        return self

    def transform(self,x):


        x["Education Level"]=self.ord.transform(x[["Education Level"]])
        encode=self.ohe.transform(x[["Job Title"]])
        encode_df=pd.DataFrame(encode,columns=ohe.get_feature_names_out(["Job Title"]))
        encode_df.drop("Job Title_accountant",axis=1,inplace=True)
        x=pd.concat([x.drop("Job Title",axis=1).reset_index(drop=True),encode_df],axis=1)
        return x