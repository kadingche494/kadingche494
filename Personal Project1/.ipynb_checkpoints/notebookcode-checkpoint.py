#!/usr/bin/env python
# coding: utf-8

# # A2: NeuralNetwork Class

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Requirements" data-toc-modified-id="Requirements-1">Requirements</a></span></li><li><span><a href="#Code-for-NeuralNetwork-Class" data-toc-modified-id="Code-for-NeuralNetwork-Class-2">Code for <code>NeuralNetwork</code> Class</a></span></li><li><span><a href="#Example-Results" data-toc-modified-id="Example-Results-3">Example Results</a></span></li><li><span><a href="#Application-to-Boston-Housing-Data" data-toc-modified-id="Application-to-Boston-Housing-Data-4">Application to Boston Housing Data</a></span></li></ul></div>

# ## Requirements

# In this assignment, you will complete the implementation of the `NeuralNetwork` class, starting with the code included in the next code cell.  Your implementation must meet the requirements described in the doc-strings.
# 
# Download [optimizers.tar](https://www.cs.colostate.edu/~anderson/cs545/notebooks/optimizers.tar) and extract `optimizers.py` for use in this assignment.
# 
# Then apply your `NeuralNetwork` class to the problem of predicting the value of houses in Boston as described below.

# ## Code for `NeuralNetwork` Class

# In[7]:


get_ipython().run_cell_magic('writefile', 'neuralnetwork.py', '\nimport numpy as np\nimport optimizers\n\n\nclass NeuralNetwork():\n    """\n    A class that represents a neural network for nonlinear regression\n\n    Attributes\n    ----------\n    n_inputs : int\n        The number of values in each sample\n    n_hidden_units_by_layers: list of ints, or empty\n        The number of units in each hidden layer.\n        Its length specifies the number of hidden layers.\n    n_outputs: int\n        The number of units in output layer\n    all_weights : one-dimensional numpy array\n        Contains all weights of the network as a vector\n    Ws : list of two-dimensional numpy arrays\n        Contains matrices of weights in each layer,\n        as views into all_weights\n    all_gradients : one-dimensional numpy array\n        Contains all gradients of mean square error with\n        respect to each weight in the network as a vector\n    Grads : list of two-dimensional numpy arrays\n        Contains matrices of gradients weights in each layer,\n        as views into all_gradients\n    total_epochs : int\n        Total number of epochs trained so far\n    error_trace : list\n        Mean square error (standardized) after each epoch\n    X_means : one-dimensional numpy array\n        Means of the components, or features, across samples\n    X_stds : one-dimensional numpy array\n        Standard deviations of the components, or features, across samples\n    T_means : one-dimensional numpy array\n        Means of the components of the targets, across samples\n    T_stds : one-dimensional numpy array\n        Standard deviations of the components of the targets, across samples\n        \n    Methods\n    -------\n    make_weights_and_views(shapes)\n        Creates all initial weights and views for each layer\n\n    train(X, T, n_epochs, method=\'sgd\', learning_rate=None, verbose=True)\n        Trains the network using samples by rows in X and T\n\n    use(X)\n        Applies network to inputs X and returns network\'s output\n    """\n\n    def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):\n        """Creates a neural network with the given structure\n\n        Parameters\n        ----------\n        n_inputs : int\n            The number of values in each sample\n        n_hidden_units_by_layers : list of ints, or empty\n            The number of units in each hidden layer.\n            Its length specifies the number of hidden layers.\n        n_outputs : int\n            The number of units in output layer\n\n        Returns\n        -------\n        NeuralNetwork object\n        """\n        self.n_inputs = 0\n        self.n_hidden_units_by_layers = []\n        self.n_outputs = 0\n        # Assign attribute values. Set self.X_means to None to indicate\n        # that standardization parameters have not been calculated.\n        # ....\n        self.n_inputs = n_inputs\n        self.n_hidden_units_by_layers = n_hidden_units_by_layers\n        self.n_outputs = n_outputs\n        self.all_weights = []\n        self.Ws = []\n        self.all_gradients = []\n        self.Grads = []\n        self.total_epochs = 0\n        self.error_trace = []\n        self.X_stds = None\n        self.T_means= None\n        self.T_stds = None\n        self.X_means = None\n        self.shapes=[]\n\n        # Build list of shapes for weight matrices in each layer\n        # ...\n        last_c = n_inputs \n        global n_hidden\n        n_hidden = len(n_hidden_units_by_layers)\n        shapes = np.random.uniform(-1, 1, size=(n_hidden + 1, 2)) / np.sqrt(n_hidden)\n        shapes = shapes.astype(int)\n        self.shapes = shapes\n        for temp_1 in range(0,n_hidden,1):\n            t_u = n_hidden_units_by_layers[temp_1]\n            shapes[temp_1,:] = (last_c + 1,t_u)\n            last_c = t_u\n        shapes [n_hidden,:] = (last_c + 1, n_outputs)\n        # Call make_weights_and_views to create all_weights and Ws\n        # ...\n        \n        self.all_weights,self.Ws= self.make_weights_and_views(shapes)\n        \n        # Call make_weights_and_views to create all_gradients and Grads\n        \n        self.all_gradients,self.Grads=self.make_weights_and_views(shapes)\n\n    def make_weights_and_views(self, shapes):\n        """Creates vector of all weights and views for each layer\n\n        Parameters\n        ----------\n        shapes : list of pairs of ints\n            Each pair is number of rows and columns of weights in each layer\n\n        Returns\n        -------\n        Vector of all weights, and list of views into this vector for each layer\n        """\n\n        # Create one-dimensional numpy array of all weights with random initial values\n        #  ...\n        \n        t_u = shapes.shape[0]\n        global sum_u\n        sum_u = 0\n        \n        for temp_r in range(t_u):\n            sum_u += shapes[temp_r,0]*shapes[temp_r,1]\n    \n        all_weights = np.random.uniform(-1, 1, size=(sum_u)) / np.sqrt(sum_u)\n       \n        # Build list of views by reshaping corresponding elements\n        # from vector of all weights into correct shape for each layer. \n        \n        Ws = []\n        point = 0\n        tt = 0\n        for count in range(0,t_u,1):\n            tt = shapes[count,0]*shapes[count,1]\n            Ws_t = all_weights[point:(point+tt):1].reshape(shapes[count,0],shapes[count,1])\n            Ws.append(Ws_t)\n            point += tt\n        return all_weights, Ws\n         \n    def __repr__(self):\n        return f\'NeuralNetwork({self.n_inputs}, \' + \\\n            f\'{self.n_hidden_units_by_layers}, {self.n_outputs})\'\n\n    def __str__(self):\n        s = self.__repr__()\n        if self.total_epochs > 0:\n            s += f\'\\n Trained for {self.total_epochs} epochs.\'\n            s += f\'\\n Final standardized training error {self.error_trace[-1]:.4g}.\'\n        return s\n \n    def train(self, X, T, n_epochs, method=\'sgd\', learning_rate=None, verbose=True):\n        """Updates the weights \n\n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  x  number of input components\n        T : two-dimensional numpy array\n            number of samples  x  number of output components\n        n_epochs : int\n            Number of passes to take through all samples\n        method : str\n            \'sgd\', \'adam\', or \'scg\'\n        learning_rate : float\n            Controls the step size of each update, only for sgd and adam\n        verbose: boolean\n            If True, progress is shown with print statements\n        """\n\n        # Calculate and assign standardization parameters\n        # ...\n        self.X_means = X.mean(axis=0)\n        self.X_stds = X.std(axis=0)\n        self.T_means = T.mean(axis=0)\n        self.T_stds = T.std(axis=0)\n    \n        # Standardize X and T\n        # ...\n        \n        Xs = (X - self.X_means) / self.X_stds\n        Ts = (T - self.T_means) / self.T_stds\n        \n        frag = []\n        frag.append(Xs)\n        frag.append(Ts)\n        \n        # Instantiate Optimizers object by giving it vector of all weights\n        optimizer = optimizers.Optimizers(self.all_weights)\n\n        error_convert_f = lambda err: (np.sqrt(err) * self.T_stds)[0]\n        \n        # Call the requested optimizer method to train the weights.\n        error_trace = []\n        nestrov = False\n        callback_f = None\n        if method == \'sgd\':\n            error_trace = optimizer.sgd(self.error_f, self.gradient_f,frag, n_epochs, learning_rate, verbose ,error_convert_f, nestrov, callback_f)\n           \n        elif method == \'adam\':\n            error_trace = optimizer.adam(self.error_f, self.gradient_f,frag, n_epochs, learning_rate, verbose,error_convert_f, callback_f)\n           \n        elif method == \'scg\':\n            error_trace = optimizer.scg(self.error_f, self.gradient_f,frag, n_epochs, error_convert_f,verbose, callback_f)\n        else:\n            raise Exception("method must be \'sgd\', \'adam\', or \'scg\'")\n\n        self.total_epochs += len(error_trace)\n        self.error_trace += error_trace\n        \n        # Return neural network object to allow applying other methods\n        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)\n        self._forward(X)\n        return self\n\n    def _forward(self, X):\n        """Calculate outputs of each layer given inputs in X\n        \n        Parameters\n        ----------\n        X : input samples, standardized\n\n        Returns\n        -------\n        Outputs of all layers as list\n        """\n        # unpack self.all_weights to self.Ws\n        i=0\n        for layerI in range(n_hidden):\n            shapeX=self.shapes[layerI,0]\n            shapeY=self.shapes[layerI,1]\n            self.Ws[layerI]=self.all_weights[i:i+shapeX*shapeY].reshape(shapeX,shapeY)\n            i+=shapeX*shapeY\n        self.Ys = [X]\n        # Append output of each layer to list in self.Ys, then return it.\n        \n        temp = self.addones(X)\n        \n        for multi_t in range (n_hidden + 1):\n            temp =temp @ self.Ws[multi_t]\n            if multi_t < n_hidden:\n                temp = np.tanh(temp)\n            self.Ys.append(temp)\n            temp = np.insert(temp, 0, 1, axis=1)\n            \n        return self.Ys\n\n    # Function to be minimized by optimizer method, mean squared error\n    def error_f(self, X, T):\n        """Calculate output of net and its mean squared error \n\n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  x  number of input components\n        T : two-dimensional numpy array\n            number of samples  x  number of output components\n\n        Returns\n        -------\n        Mean square error as scalar float that is the mean\n        square error over all samples\n        """\n        # Call _forward, calculate mean square error and return it.\n        \n        Y_t= self._forward(X)\n        Y = Y_t[-1]\n        error = (T - Y) * self.T_stds + self.T_means\n        error_f_value = np.sqrt(np.mean(error ** 2))\n        return error_f_value\n\n    # Gradient of function to be minimized for use by optimizer method\n    \n    def addones(self, temp1):\n        return np.insert(temp1, 0, 1, axis=1)\n        \n    def gradient_f(self, X, T):\n        """Returns gradient wrt all weights. Assumes _forward already called.\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  x  number of input components\n        T : two-dimensional numpy array\n            number of samples  x  number of output components\n\n        Returns\n        -------\n        Vector of gradients of mean square error wrt all weights\n        """\n\n        # Assumes forward_pass just called with layer outputs saved in self.Ys.\n        n_samples = X.shape[0]\n        n_outputs = T.shape[1]\n        Y = self._forward(X)\n        \n        D = -(T - Y[-1])/ n_samples*n_outputs\n        \n        all_gradients =np.random.uniform(-1, 1, size=(sum_u)) / np.sqrt(sum_u)\n        # Step backwards through the layers to back-propagate the error (D)\n        for layeri in range(n_hidden, -1, -1):\n            if layeri > 0:\n                self.Grads[layeri] =  self.addones(self.Ys[layeri]).T @ D\n                \n                D = (D @ self.Ws[layeri][1:, :].T) * (1 - self.Ys[layeri]**2)\n            else:\n                self.Grads[layeri] = self.addones(X).T @ D\n            \n        columns = 0        \n        for count_1 in range(n_hidden):\n            i,j =self.Grads[count_1].shape\n            all_gradients[columns:columns + i*j] = self.Grads[count_1].reshape(1,-1)\n            columns += i * j \n        \n        self.all_gradients = all_gradients\n        \n        return self.all_gradients\n\n    def use(self, X):\n        """Return the output of the network for input samples as rows in X\n\n        Parameters\n        ----------\n        X : two-dimensional numpy array\n            number of samples  x  number of input components, unstandardized\n\n        Returns\n        -------\n        Output of neural network, unstandardized, as numpy array\n        of shape  number of samples  x  number of outputs\n        """\n\n        # Standardize X\n        Xs = (X - X.mean(axis=0)) / X.std(axis=0)\n        Y_t = self._forward(Xs)\n        Y = Y_t[-1]\n        # Unstandardize output Y beforereturning it\n        return Y * self.T_stds + self.T_means\n\n    def get_error_trace(self):\n        """Returns list of standardized mean square error for each epoch"""\n        return self.error_trace')


# ## Example Results

# Here we test the `NeuralNetwork` class with some simple data.  
# 

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import neuralnetwork as nn

X = np.arange(-2, 2, 0.05).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)

# Just use first 5 samples
X = X[:5, :]
T = T[:5, :]

errors = []
# n_epochs = 1000
n_epochs = 10

method_rhos = [('sgd', 0.01),
               ('adam', 0.005),
               ('scg', None)]

for method, rho in method_rhos:
    
    print('\n=========================================')
    print(f'method is {method} and rho is {rho}')
    print('=========================================\n')

    nnet = nn.NeuralNetwork(X.shape[1], [2, 2], 1)
    
    # Set all weights here to allow comparison of your calculations
    # Must use [:] to overwrite values in all_weights.
    # Without [:], new array is assigned to self.all_weights, so self.Ws no longer refer to same memory
    nnet.all_weights[:] = np.arange(len(nnet.all_weights)) * 0.001
    
    nnet.train(X, T, n_epochs, method=method, learning_rate=rho)
    Y = nnet.use(X)
    plt.plot(X, Y, 'o-', label='Model ' + method)
    errors.append(nnet.get_error_trace())

plt.plot(X, T, 'o', label='Train')
plt.xlabel('X')
plt.ylabel('T or Y')
plt.legend();


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import neuralnetwork as nn

X = np.arange(-2, 2, 0.05).reshape(-1, 1)
T = np.sin(X) * np.sin(X * 10)

errors = []
n_epochs = 1000
method_rhos = [('sgd', 0.01),
               ('adam', 0.005),
               ('scg', None)]

for method, rho in method_rhos:
    nnet = nn.NeuralNetwork(X.shape[1], [10, 10,8,9,10], 1)
    nnet.train(X, T, 50000, method=method, learning_rate=rho)
    Y = nnet.use(X)
    plt.plot(X, Y, 'o-', label='Model ' + method)
    errors.append(nnet.get_error_trace())

plt.plot(X, T, 'o', label='Train')
plt.xlabel('X')
plt.ylabel('T or Y')
plt.legend();


# In[5]:


plt.figure(2)
plt.clf()
for error_trace in errors:
    plt.plot(error_trace)
plt.xlabel('Epoch')
plt.ylabel('Standardized error')
plt.legend([mr[0] for mr in method_rhos]);


# Your results will not be the same, but your code should complete and make plots somewhat similar to these.

# ## Application to Boston Housing Data

# Download data from [Boston House Data at Kaggle](https://www.kaggle.com/fedesoriano/the-boston-houseprice-data). Read it into python using the `pandas.read_csv` function.  Assign the first 13 columns as inputs to `X` and the final column as target values to `T`.  Make sure `T` is two-dimensional.

# Before training your neural networks, partition the data into training and testing partitions, as shown here.

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd # for reading csv file
from IPython.display import display, clear_output  # for animations later in this notebook
import neuralnetwork as nn


# In[ ]:





# In[5]:


X = pd.read_csv('boston.csv',usecols=range(13),index_col = False)
T = pd.read_csv('boston.csv',usecols=[13],index_col = False)


# In[6]:


X = np.array(X)
T = np.array(T).reshape((-1, 1))


# In[7]:


def partition(X, T, train_fraction):
    n_samples = X.shape[0]
    rows = np.arange(n_samples)
    np.random.shuffle(rows)
    
    n_train = round(n_samples * train_fraction)
    
    Xtrain = X[rows[:n_train], :]
    Ttrain = T[rows[:n_train], :]
    Xtest = X[rows[n_train:], :]
    Ttest = T[rows[n_train:], :]
    
    return Xtrain, Ttrain, Xtest, Ttest
    
def rmse(T, Y):
    return np.sqrt(np.mean((T - Y)**2))


# In[8]:


# Assuming you have assigned `X` and `T` correctly.

Xtrain, Ttrain, Xtest, Ttest = partition(X, T, 0.8)  


# In[13]:


errors = []

method_rhos = [('sgd', 0.01),
               ('adam', 0.005),
               ('scg', None)]
n_inputs = Xtrain.shape[1]
n_outputs = Ttrain.shape[1]

for method, rho in method_rhos:
    nnet = nn.NeuralNetwork(n_inputs, [10, 10], n_outputs)
    nnet.train(Xtrain, Ttrain, 50000, method=method, learning_rate=rho)
    Ytrain = nnet.use(Xtrain)
    Ytest = nnet.use(Xtest)
    error_rmse = rmse(Ttest, Ytest)
    print ('rmse for test = {:.10f}'.format(error_rmse))
    plt.plot(Xtrain, Ytrain, 'o-', label='Model ' + method)
    plt.plot(Xtest, Ytest, 'o-', label='Model ' + method)
    errors.append(nnet.get_error_trace())
plt.xlabel('X')
plt.ylabel('T or Y')
plt.legend();


# In[12]:


plt.plot(Xtrain, Ttrain, 'o-', Xtest, Ttest, 'o-', Xtest, Ytest, 'o-')
plt.xlim(-10, 10)
plt.legend(('Training', 'Testing', 'Model'), loc='upper left')


# In[ ]:


#After I running this boston housing code, I believe the  accurate
#results will denpend on suitable rho and n_opchs. And my results 
#are not good enough and need to find some better parameter for
#my training function. By the amount of Data I used to train,
# this result is good but not good enough.


# Write and run code using your `NeuralNetwork` class to model the Boston housing data. Experiment with all three optimization methods and a variety of neural network structures (numbers of hidden layer and units), learning rates, and numbers of epochs. Show results for at least three different network structures, learning rates, and numbers of epochs for each method.  Show your results using print statements that include the method, network structure, number of epochs, learning rate, and RMSE on training data and RMSE on testing data.
# 
# Try to find good values for the RMSE on testing data.  Discuss your results, including how good you think the RMSE values are by considering the range of house values given in the data. 

# # Grading
# 
# Your notebook will be run and graded automatically. Test this grading process by first downloading [A2grader.tar](http://www.cs.colostate.edu/~anderson/cs545/notebooks/A2grader.tar) and extract `A2grader.py` from it. Run the code in the following cell to demonstrate an example grading session.  The remaining 20 points will be based on your discussion of this assignment.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.  
# 
# For the grading script to run correctly, you must first name this notebook as 'Lastname-A2.ipynb' with 'Lastname' being your last name, and then save this notebook.

# In[6]:


get_ipython().run_line_magic('run', '-i A2grader.py')


# # Extra Credit
# 
# Apply your multilayer neural network code to a regression problem using data that you choose 
# from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php). Pick a dataset that
# is listed as being appropriate for regression.

# In[ ]:




