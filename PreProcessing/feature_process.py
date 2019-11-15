import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def std_PCA(**argv):
    scaler = StandardScaler()
    pca = PCA(**argv)
    pipeline = Pipeline([('scaler',scaler),
                         ('pca',pca)])
    return pipeline

if __name__ == "__main__":
    with open("all_data", 'rb') as f:
     all_data = pickle.load(f)

    polynomial_features = PolynomialFeatures(degree=3,include_bias=False)
    all_dataselect = all_data.drop(labels=['simple_rat_index', 'soil_reg_veg_index', 'sr', 'nri', 'tpvi', 'green_red_ndvi'], axis=1).sample(n=50000, axis=0)
    pca = std_PCA(n_components=3)  # PCA降维为两个特征值
    y = np.array(all_dataselect['label']).ravel()
    X = all_dataselect.drop('label', axis=1)
    X_pca = pca.fit_transform(all_dataselect.drop('label', axis=1))
    X_pca5w3d=polynomial_features.fit_transform(X_pca)
    X_train53, X_test53, y_train53, y_test53 = train_test_split(X_pca5w3d, y, test_size=0.1, random_state=42)
    with open("X_train53", 'wb') as f:
        pickle.dump(X_train53, f)
    with open("X_test53", 'wb') as f:
        pickle.dump(X_test53, f)
    with open("y_train53", 'wb') as f:
        pickle.dump(y_train53, f)
    with open("y_test53", 'wb') as f:
        pickle.dump(y_test53, f)