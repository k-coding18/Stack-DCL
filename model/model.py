import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def cosine_distance_matrix(X, sigma=1.0):
    return cdist(X, X, metric='cosine')


def gaussian_kernel_distance_matrix(X, sigma=1.0):
    euclidean_dist = cdist(X, X, metric='euclidean')
    return 1 - np.exp(-(euclidean_dist ** 2) / (2.0 * (sigma ** 2)))

class LogReg:
    def __init__(self, C=0.05, max_iter=1000, random_state=33):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                solver='liblinear',
                C=C,
                penalty='l2',
                class_weight=None,
                max_iter=max_iter,
                random_state=random_state
            ))
        ])
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)


class DAWRF(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=200, k=4, random_state=33, flag=1):
        self.n_estimators = n_estimators
        self.k = k
        self.flag = flag
        self.random_state = random_state
        self.rf = None
        self.classes_ = None
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train, sigma=1.0):
        self.X_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(y_train)
        final_weights = self.calculate_density_weights(X_train, self.k, sigma)
        self.rf = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        self.rf.fit(X_train, y_train, sample_weight=final_weights)
        return self

    def predict(self, X):
        y_pred = self.rf.predict(X)
        return y_pred

    def predict_proba(self, X):
        return self.rf.predict_proba(X)

    def calculate_density_weights(self, X, k, sigma=1.0):
        distance_matrix = gaussian_kernel_distance_matrix(X, sigma)
        index = np.argsort(distance_matrix)[:, 1:k + 1]
        distances = np.take_along_axis(distance_matrix, index, axis=1)
        density = distances[:, 1:].sum(axis=1)
        density_weights = 1 / (density + 1e-5)  # 防止除零
        density_weights = (density_weights - np.min(density_weights)) / (
                    np.max(density_weights) - np.min(density_weights) + 1e-5)
        return density_weights


class CHKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=4):
        self.n_neighbors = n_neighbors
        self.classes_ = None
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(y_train)  # 确定所有类别
        return self

    def _compute_local_hyperplane(self, neighbors, weights):
        return np.dot(weights, neighbors)  # 超平面为邻居点加权求和

    def _distance_to_hyperplane(self, x, hyperplane):
        dot_product = np.dot(x, hyperplane)

        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(hyperplane)

        cosine_similarity = dot_product / (norm_x * norm_y)

        lam = 1
        p = 2
        cosine_distance = np.exp(lam * (1 - cosine_similarity) ** (p / 2))
        return cosine_distance

    def predict(self, X_test):
        predictions = []
        for i, x in enumerate(X_test):
            min_distance = float('inf')
            predicted_class = None
            for cls in self.classes_:
                class_indices = np.where(self.y_train == cls)[0]
                class_samples = self.X_train[class_indices]
                matrix = cosine_distance_matrix(np.vstack([class_samples, x]))
                min_index = np.argsort(matrix[-1, :class_samples.shape[0]])[:self.n_neighbors]
                distances = matrix[-1, min_index]
                neighbors = class_samples[min_index]

                sigma = 1.0
                weights = np.exp(-distances ** 2 / (2 * sigma ** 2))

                hyperplane = self._compute_local_hyperplane(neighbors, weights)
                distance = self._distance_to_hyperplane(x, hyperplane)

                if distance < min_distance:
                    min_distance = distance
                    predicted_class = cls

            predictions.append(predicted_class)
        return np.array(predictions)

    def predict_proba(self, X_test):
        probas = []
        for i, x in enumerate(X_test):
            class_scores = []
            for cls in self.classes_:
                class_indices = np.where(self.y_train == cls)[0]
                class_samples = self.X_train[class_indices]
                matrix = cosine_distance_matrix(np.vstack([class_samples, x]))
                min_index = np.argsort(matrix[-1, :class_samples.shape[0]])[:self.n_neighbors]
                distances = matrix[-1, min_index]
                neighbors = class_samples[min_index]

                sigma = 1.0
                weights = np.exp(-distances ** 2 / (2 * sigma ** 2))

                hyperplane = self._compute_local_hyperplane(neighbors, weights)
                distance = self._distance_to_hyperplane(x, hyperplane)
                class_scores.append(distance)

            inv_scores = 1 / (np.array(class_scores) + 1e-8)
            prob = inv_scores / np.sum(inv_scores)
            probas.append(prob)
        proba_array = np.array(probas)
        proba_array = proba_array.astype(np.float64)
        proba_array /= np.sum(proba_array, axis=1, keepdims=True) + 1e-8

        return proba_array


class Stack_DCL(BaseEstimator, ClassifierMixin):
    def __init__(self, stack_m='predict'):
        chknn = CHKNN(n_neighbors=4)
        darf = DAWRF(n_estimators=200, k=4, random_state=42, flag=1)
        logreg = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                solver='liblinear',
                C=0.05,
                penalty='l2',
                class_weight=None,
                max_iter=1000,
                random_state=33
            ))
        ])
        self.stacking_model = StackingClassifier(
            estimators=[('chknn', chknn), ('darf', darf), ('logreg', logreg)],
            final_estimator=LogisticRegression(),
            n_jobs=-1,
            stack_method=stack_m,
        )

    def fit(self, X, y):
        self.stacking_model.fit(X, y)
        return self

    def predict(self, X):
        return self.stacking_model.predict(X)
    
    def predict_proba(self, X):
        return self.stacking_model.predict_proba(X)
    

class Stack_DC(BaseEstimator, ClassifierMixin):
    def __init__(self, stack_m='predict', flag=1):
        chknn = CHKNN(n_neighbors=4)
        darf = DAWRF(n_estimators=200, k=4, random_state=42, flag=1)
        self.stacking_model = StackingClassifier(
            estimators=[('chknn', chknn), ('darf', darf)],
            final_estimator=LogisticRegression(),
            n_jobs=-1,
            stack_method=stack_m,
        )

    def fit(self, X, y):
        self.stacking_model.fit(X, y)
        return self

    def predict(self, X):
        return self.stacking_model.predict(X)
    
    def predict_proba(self, X):
        return self.stacking_model.predict_proba(X)

    
class Stack_CL(BaseEstimator, ClassifierMixin):
    def __init__(self, stack_m='predict', flag=1):
        chknn = CHKNN(n_neighbors=4)
        logreg = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                solver='liblinear',
                C=0.05,
                penalty='l2',
                class_weight=None,
                max_iter=1000,
                random_state=33
            ))
        ])
        self.stacking_model = StackingClassifier(
            estimators=[('chknn', chknn), ('logreg', logreg)],
            final_estimator=LogisticRegression(),
            n_jobs=-1,
            stack_method=stack_m,
        )

    def fit(self, X, y):
        self.stacking_model.fit(X, y)
        return self

    def predict(self, X):
        return self.stacking_model.predict(X)
    
    def predict_proba(self, X):
        return self.stacking_model.predict_proba(X)
    
    
class Stack_DL(BaseEstimator, ClassifierMixin):
    def __init__(self, stack_m='predict', flag=1):
        darf = DAWRF(n_estimators=200, k=4, random_state=42, flag=1)
        logreg = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                solver='liblinear',
                C=0.05,
                penalty='l2',
                class_weight=None,
                max_iter=1000,
                random_state=33
            ))
        ])
        self.stacking_model = StackingClassifier(
            estimators=[('darf', darf), ('logreg', logreg)],
            final_estimator=LogisticRegression(),
            n_jobs=-1,
            stack_method=stack_m,
        )
      
    def fit(self, X, y):
        self.stacking_model.fit(X, y)
        return self

    def predict(self, X):
        return self.stacking_model.predict(X)
    
    def predict_proba(self, X):
        return self.stacking_model.predict_proba(X)