from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.optimize import minimize

class collaborative_filtering(object):

    def __init__(self, num_features:int = 2, regularization_param:float = 0.0, max_iter:int = 500, random_state:int=None):
        '''
            PARAMTERS
            -----------------
            num_features: the number of features
            regularization_parameter: the regularization parameters alpha
            max_iter: the number of iterations to 
        '''
        self._num_features = num_features
        self._regularization_param = regularization_param
        self._max_iter = max_iter
        self._random_state = random_state

    def _initialize_thetas(self, rows:int, cols:int):
        '''
            DESCRIPTION
            ------------
            initializes the parameters with random values between 0 and 1 and unrolls them into a 1 dimenional vector
        '''
        np.random.seed(self._random_state)
        thetas1 = np.random.rand(rows, self._num_features) # X

        np.random.seed(self._random_state)
        thetas2 = np.random.rand(self._num_features, cols) # theta
        
        self._thetas = np.concatenate((thetas1.flatten(), thetas2.flatten()))

    def _reshape_thetas(self, thetas:np.ndarray, rows:int, cols:int):
        thetas1 = thetas[:rows * self._num_features].reshape(rows, self._num_features)
        thetas2 = thetas[rows * self._num_features:].reshape(self._num_features, cols)
        return thetas1, thetas2

    def _gradient(self, thetas:np.ndarray, Y:np.ndarray, R:np.ndarray):
        '''
            calculates the partial derivatives with respect to 
            each the parameters and unrolls them into a 1 dimensional vector
        '''
        rows, cols = Y.shape
        thetas1, thetas2 = self._reshape_thetas(thetas, rows, cols)
 
        predictions = np.matmul(thetas1, thetas2)
        prediction_error = (predictions - Y) * R

        thetas1_gradient = np.matmul(prediction_error, thetas2.T)
        thetas2_gradient = np.matmul(prediction_error.T, thetas1).T

        if self._regularization_param > 0:
            thetas1_gradient += thetas1 * self._regularization_param
            thetas2_gradient += thetas2 * self._regularization_param

        return np.concatenate((thetas1_gradient.flatten(), thetas2_gradient.flatten()))

    def _cost(self, thetas:np.ndarray, Y:np.ndarray, R:np.ndarray):
        '''
            calculates the value of the cost function
        '''
        rows, cols = Y.shape
        thetas1, thetas2 = self._reshape_thetas(thetas, rows, cols)

        # calculate predictions
        predictions = np.matmul(thetas1, thetas2)

        # square errors
        mse = np.power(predictions - Y, 2)

        # only take into account those where R = 1
        relevant_mse = np.sum(mse * R)

        # add regularization if provided
        if self._regularization_param > 0:
            relevant_mse += self._regularization_param * np.sum(np.power(thetas, 2))

        relevant_mse *= 0.5
        return relevant_mse

    def fit(self, Y:np.ndarray, R:np.ndarray):
        '''
            PARAMETERS
            ---------
            Y: ratings matrix
            R: matrix of 1's (rating provided) or 0's (rating not provided)
        '''
        self._R = R
        self._Y = Y

        # initialize thetas values
        self._rows, self._cols = Y.shape
        self._initialize_thetas(self._rows, self._cols)

        # store the means for each row, take into account only values for which R = 1
        self._means = np.nanmean(np.where(R, Y, np.nan), axis=1, keepdims=True)

        # subtract the mean from Y
        # the mean is added back when doing predictions
        # this helps us deal with cases when a user has not rated any movie yet
        normalized_Y = np.where(R, Y - self._means, 0)

        # optimize
        optimized = minimize(fun=self._cost, x0=self._thetas, args=(normalized_Y, R), method='TNC', jac=self._gradient, options={'maxiter':self._max_iter, 'disp':True})
        self._thetas = optimized.x

    def predict(self, iid:int, uid:int):
        '''
            DESCRIPTION
            ------------
            Returns the predicted rating that user uid gives to item iid

            PARAMETERS
            -------------
            iid: index of the item to predict the rating for. Index is zero based
            uid: index of the user who rates item iid. Index is zero based
        '''
        thetas1, thetas2 = self._reshape_thetas(self._thetas, self._rows, self._cols)
        return np.matmul(thetas1[iid, :], thetas2[:, uid]) + self._means[iid]
    
    def recommended_items(self, uid:int):
        '''
            DESCRIPTION
            -----------
            Returns the sorted indexes (descending order) of the predicted ratings user uuid would give to items she has yet to rate
        '''
        # reshape thetas
        thetas1, thetas2 = self._reshape_thetas(self._thetas, self._rows, self._cols)

        # predict ratings for the ith user
        predictions = np.where(self._R[:, [uid]], np.nan, np.matmul(thetas1, thetas2[:, [uid]]) + self._means)

        # total number of items user has not rated
        total_user_unrated_items = np.sum(self._R[:, uid] == 0)

        return np.argsort(-predictions, axis=0)[:total_user_unrated_items]
    
    def compute_similarities(self, users_items:int = 0):
        '''
            DESCRIPTION
            -----------
            computes the cosine similarities matrix of either users or items

            PARAMETERS
            -----------
            users_items: 0 if you want to return the cosine similarities of the items, 1 if you want to return the cosine similarities of the users
        '''

        # reshape thetas
        thetas1, thetas2 = self._reshape_thetas(self._thetas, self._rows, self._cols)

        if users_items == 0:
            return cosine_similarity(thetas1)
        elif users_items == 1:
            return cosine_similarity(thetas2.T)
        else:
            raise ValueError('users_items must be either 0 (items) or 1 (users)')
