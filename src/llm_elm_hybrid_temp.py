    def fit(self, X, y):
        """Fit the ELM model using least squares for output weights."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        
        # Compute hidden layer output
        hidden_input = np.dot(X, self.hidden_weights) + self.hidden_bias
        hidden_output = self.activation(hidden_input)
        
        # Solve for output weights using least squares
        try:
            self.output_weights = np.linalg.pinv(hidden_output) @ y
        except:
            # Fallback to normal equations
            self.output_weights = np.linalg.solve(
                hidden_output.T @ hidden_output + 1e-6 * np.eye(hidden_output.shape[1]),
                hidden_output.T @ y
            )
    
    def predict(self, X):
        """Make predictions using the trained ELM."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Compute hidden layer output
        hidden_input = np.dot(X, self.hidden_weights) + self.hidden_bias
        hidden_output = self.activation(hidden_input)
        
        # Compute output
        output = np.dot(hidden_output, self.output_weights)
        
        return output
