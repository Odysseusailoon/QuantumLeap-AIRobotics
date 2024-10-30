import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class GaussianProcess:
    def __init__(self, kernel='rbf'):
        self.kernel = kernel
        
    def rbf_kernel(self, x1, x2, length_scale=1.0):
        distance = (x1 - x2)**2
        return np.exp(-distance / (2 * length_scale**2))
        
    def fit(self, X, y):
        """
        Fit the Gaussian Process model
        X: input points (n_samples, n_features)
        y: target values (n_samples,)
        """
        self.X_train = X.reshape(-1, 1)  # Store training data
        self.y_train = y
        
        # Compute kernel matrix
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.rbf_kernel(X[i], X[j])
        
        # Add small noise to diagonal for numerical stability
        K += 1e-8 * np.eye(n)
        
        # Store for prediction
        self.K = K
        # Compute inverse of K
        self.K_inv = np.linalg.inv(K)

    def predict(self, X_test):
        """
        Make predictions at new points
        X_test: new input points (n_new_samples,)
        Returns: mean and std of predictions
        """
        # Compute kernel between new points and training points
        k_star = np.array([[self.rbf_kernel(x1, x2) for x2 in self.X_train.flatten()] 
                           for x1 in X_test])
        print(f"k_star shape: {k_star.shape}")
        print(f"K_inv shape: {self.K_inv.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        
        # Compute mean prediction
        mean = k_star @ self.K_inv @ self.y_train
        
        # Compute kernel between new points
        K_star_star = np.array([[self.rbf_kernel(x1, x2) for x2 in X_test] 
                               for x1 in X_test])
        
        # Compute variance
        var = K_star_star - k_star @ self.K_inv @ k_star.T
        std = np.sqrt(np.diag(var))
        
        return mean, std

# Test code
if __name__ == "__main__":
    # Test data
    X = np.linspace(0, 10, 100)
    y = np.sin(X) + np.random.normal(0, 0.1, 100)
    # Generate training data
    X = np.linspace(0, 10, 20)  # Fewer points for faster computation
    y = np.sin(X) + np.random.normal(0, 0.1, 20)
    
    # Fit GP
    gp = GaussianProcess()
    gp.fit(X, y)
    
    # Generate test points
    X_test = np.linspace(0, 10, 100)
    mean, std = gp.predict(X_test)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Training data')
    plt.plot(X_test, mean, 'r-', label='Mean prediction')
    plt.fill_between(X_test, mean - 2*std, mean + 2*std, color='red', alpha=0.2, label='95% confidence interval')
    plt.title('Gaussian Process Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    gp.fit(X, y) 
    plt.scatter(X, y, label='Data')
    plt.title('Generated Test Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
