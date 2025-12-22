from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from preprocessing.BaseTransformer import BaseTransformer

class NumericScaler(BaseTransformer):

    def __init__(self, method="standard"):
        self.method = method
        self.scaler = None
    
    def fit(self, X):
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        elif self.method == "maxabs":
            self.scaler = MaxAbsScaler()
        elif self.method == "quantile_uniform":
            self.scaler = QuantileTransformer(output_distribution='uniform', random_state=42)
        elif self.method == "quantile_normal":
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        elif self.method == "yeo_johnson":
            self.scaler = PowerTransformer(method='yeo-johnson')
        elif self.method == "boxcox":
            self.scaler = PowerTransformer(method='box-cox')
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
        
        self.scaler.fit(X)
    
    def transform(self, X):
        return self.scaler.transform(X)
