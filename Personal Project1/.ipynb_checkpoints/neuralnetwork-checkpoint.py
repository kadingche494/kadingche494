
import numpy as np
import optimizers as opt

class NeuralNetwork():
    """
    A class that represents a neural network for nonlinear regression

    Attributes
    ----------
    n_inputs : int
        The number of values in each sample
    n_hidden_units_by_layers: list of ints, or empty
        The number of units in each hidden layer.
        Its length specifies the number of hidden layers.
    n_outputs: int
        The number of units in output layer
    all_weights : one-dimensional numpy array
        Contains all weights of the network as a vector
    Ws : list of two-dimensional numpy arrays
        Contains matrices of weights in each layer,
        as views into all_weights
    all_gradients : one-dimensional numpy array
        Contains all gradients of mean square error with
        respect to each weight in the network as a vector
    Grads : list of two-dimensional numpy arrays
        Contains matrices of gradients weights in each layer,
        as views into all_gradients
    total_epochs : int
        Total number of epochs trained so far
    error_trace : list
        Mean square error (standardized) after each epoch
    X_means : one-dimensional numpy array
        Means of the components, or features, across samples
    X_stds : one-dimensional numpy array
        Standard deviations of the components, or features, across samples
    T_means : one-dimensional numpy array
        Means of the components of the targets, across samples
    T_stds : one-dimensional numpy array
        Standard deviations of the components of the targets, across samples
        
    Methods
    -------
    make_weights_and_views(shapes)
        Creates all initial weights and views for each layer

    train(X, T, n_epochs, method='sgd', learning_rate=None, verbose=True)
        Trains the network using samples by rows in X and T

    use(X)
        Applies network to inputs X and returns network's output
    """

    def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):
        """Creates a neural network with the given structure

        Parameters
        ----------
        n_inputs : int
            The number of values in each sample
        n_hidden_units_by_layers : list of ints, or empty
            The number of units in each hidden layer.
            Its length specifies the number of hidden layers.
        n_outputs : int
            The number of units in output layer

        Returns
        -------
        NeuralNetwork object
        """
        self.n_inputs = 0
        self.n_hidden_units_by_layers = []
        self.n_outputs = 0
        # Assign attribute values. Set self.X_means to None to indicate
        # that standardization parameters have not been calculated.
        # ....
        self.n_inputs = n_inputs
        self.n_hidden_units_by_layers = n_hidden_units_by_layers
        self.n_outputs = n_outputs
        self.all_weights = []
        self.Ws = []
        self.all_gradients = []
        self.Grads = []
        self.total_epochs = 0
        self.error_trace = []
        self.X_stds = None
        self.T_means= None
        self.T_stds = None
        self.X_means = None
  

        # Build list of shapes for weight matrices in each layer
        # ...
        last_c = n_inputs 
        global n_hidden
        n_hidden = len(n_hidden_units_by_layers)
        shapes = np.random.uniform(-1, 1, size=(n_hidden + 1, 2)) / np.sqrt(n_hidden)
        shapes = shapes.astype(int)
        for temp_1 in range(0,n_hidden,1):
            t_u = n_hidden_units_by_layers[temp_1]
            shapes[temp_1,:] = (last_c + 1,t_u)
            last_c = t_u
        shapes [n_hidden,:] = (last_c + 1, n_outputs)
            
        # Call make_weights_and_views to create all_weights and Ws
        # ...
        
        self.all_weights,self.Ws= self.make_weights_and_views(shapes)
        
        # Call make_weights_and_views to create all_gradients and Grads
        
        self.all_gradients,self.Grads=self.make_weights_and_views(shapes)

    def make_weights_and_views(self, shapes):
        """Creates vector of all weights and views for each layer

        Parameters
        ----------
        shapes : list of pairs of ints
            Each pair is number of rows and columns of weights in each layer

        Returns
        -------
        Vector of all weights, and list of views into this vector for each layer
        """

        # Create one-dimensional numpy array of all weights with random initial values
        #  ...
        
        t_u = shapes.shape[0]
        global sum_u
        sum_u = 0
        
        for temp_r in range(t_u):
            sum_u += shapes[temp_r,0]*shapes[temp_r,1]
    
        all_weights = np.random.uniform(-1, 1, size=(1,sum_u)) / np.sqrt(sum_u)
       
        # Build list of views by reshaping corresponding elements
        # from vector of all weights into correct shape for each layer. 
        
        Ws = []
        point = 0
        tt = 0
        for count in range(0,t_u,1):
            tt = shapes[count,0]*shapes[count,1]
            Ws_t = all_weights[0,point:(point+tt)].reshape(shapes[count,0],shapes[count,1])
            Ws.append(Ws_t)
            point += tt
        return all_weights, Ws
         
    def __repr__(self):
        return f'NeuralNetwork({self.n_inputs}, ' + \
            f'{self.n_hidden_units_by_layers}, {self.n_outputs})'

    def __str__(self):
        s = self.__repr__()
        if self.total_epochs > 0:
            s += f'\n Trained for {self.total_epochs} epochs.'
            s += f'\n Final standardized training error {self.error_trace[-1]:.4g}.'
        return s
 
    def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
        """Updates the weights 

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  x  number of input components
        T : two-dimensional numpy array
            number of samples  x  number of output components
        n_epochs : int
            Number of passes to take through all samples
        method : str
            'sgd', 'adam', or 'scg'
        learning_rate : float
            Controls the step size of each update, only for sgd and adam
        verbose: boolean
            If True, progress is shown with print statements
        """

        # Calculate and assign standardization parameters
        # ...
        self.X_means = X.mean(axis=0)
        self.X_stds = X.std(axis=0)
        self.T_means = T.mean(axis=0)
        self.T_stds = T.std(axis=0)
    
        # Standardize X and T
        # ...
        
        self.X = (X - self.X_means) / self.X_stds
        self.T = (T - self.T_means) / self.T_stds
        
        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = opt.Optimizers(self.all_weights)

        error_convert_f = lambda err: (np.sqrt(err) * self.T_stds)[0]
        
        # Call the requested optimizer method to train the weights.
        error_trace = []
        if method == 'sgd':
            error_trace = optimizer.sgd(self.error_f, self.gradient_f,[self.X,self.T], n_epochs, learning_rate, verbose ,error_convert_f,False,None)
           
        elif method == 'adam':
            error_trace = optimizer.adam(self.error_f, self.gradient_f,[self.X,self.T], n_epochs, learning_rate, verbose,error_convert_f, None)
           
        elif method == 'scg':
            error_trace = optimizer.scg(self.error_f, self.gradient_f,[self.X,self.T], n_epochs, error_convert_f,verbose, None)
        else:
            raise Exception("method must be 'sgd', 'adam', or 'scg'")

        self.total_epochs += len(error_trace)
        self.error_trace += error_trace
        
        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)
        self._forward(X)
        return self

    def _forward(self, X):
        """Calculate outputs of each layer given inputs in X
        
        Parameters
        ----------
        X : input samples, standardized

        Returns
        -------
        Outputs of all layers as list
        """
        self.Ys = [X]
        # Append output of each layer to list in self.Ys, then return it.
        
        temp = np.insert(X, 0, 1, axis=1) 
        
        for multi_t in range (n_hidden + 1):
            temp =temp @ self.Ws[multi_t]
            if multi < n_hiiden
                temp = np.tanh(temp)
            self.Ys.append(temp)
            temp = np.insert(temp, 0, 1, axis=1) 
            
        return self.Ys

    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        """Calculate output of net and its mean squared error 

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  x  number of input components
        T : two-dimensional numpy array
            number of samples  x  number of output components

        Returns
        -------
        Mean square error as scalar float that is the mean
        square error over all samples
        """
        # Call _forward, calculate mean square error and return it.
        Y_t = self._forward (X)
        Y = Y_t[-1]
        error = (T - Y) * (T.std(axis=0)) 
        return np.sqrt(np.mean(error ** 2))

    # Gradient of function to be minimized for use by optimizer method
    
    def addones(self, temp1):
        return np.insert(temp1, 0, 1, axis=1)
        
    def gradient_f(self, X, T):
        """Returns gradient wrt all weights. Assumes _forward already called.

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  x  number of input components
        T : two-dimensional numpy array
            number of samples  x  number of output components

        Returns
        -------
        Vector of gradients of mean square error wrt all weights
        """

        # Assumes forward_pass just called with layer outputs saved in self.Ys.
        n_samples = X.shape[0]
        n_outputs = T.shape[1]

        # D is delta matrix to be back propagated
        D = -(T - self.Ys[-1]) / (n_samples * n_outputs)
        
        all_gradients = np.random.uniform(-1, 1, size=(1, sum_u)) / np.sqrt(sum_u)
        
        # Step backwards through the layers to back-propagate the error (D)
        for layeri in range(n_hidden, -1, -1):
            self.Grads[layeri] =  self.addones(self.Ys[layeri]).T @ D
            if layeri > 0:
                D = D @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri]**2)
                
        columns = 0        
        for count_1 in range(n_hidden):
            i,j =self.Grads[count_1].shape
            all_gradients[0,columns:columns + i*j] = self.Grads[count_1].reshape(1,-1)
            columns += i * j 
        
        self.all_gradients = all_gradients
        return self.all_gradients

    def use(self, X):
        """Return the output of the network for input samples as rows in X

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  x  number of input components, unstandardized

        Returns
        -------
        Output of neural network, unstandardized, as numpy array
        of shape  number of samples  x  number of outputs
        """

        # Standardize X
        Xs = (X - X.mean(axis=0)) / X.std(axis=0)
        Y_t = self._forward(Xs)
        Y = Y_t[-1]
        # Unstandardize output Y before returning it
        return Y * Y.std(axis=0) + Y.mean(axis=0)

    def get_error_trace(self):
        """Returns list of standardized mean square error for each epoch"""
        return self.error_trace
