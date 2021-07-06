# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:51:46 2016
A programme generating texts whose word frequency is governed by
Zipf's law. 
@author: shan
"""


import random 
import bisect 
import math 
from functools import reduce


class ZipfGenerator: 
    """
    ZipfGenerator is an immutable type representing a Zipf cumulative dicstribution
    function with patameters alpha and n. 
    
    Adapted from codes copid form the flollowing online resource:
    
    http://stackoverflow.com/questions/1366984/
    generate-random-numbers-distributed-by-zipf/
    8788662#8788662

    """

    
    def __init__(self, n, alpha): 
        """Initialize a Zipf CDF.
         Paramerters
         n: int 
            n >= 0
         
         alpha: float 
            alpha >= 1
        """
        # Calculate Zeta values from 1 to n: 
        assert n >= 0 and alpha >= 1.0
        assert int(n) == n 
        self.n = n
        self.alpha = alpha
        tmp = [1. / (math.pow(float(i), alpha)) for i in range(1, n+1)] 
        zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0]) 

        # Store the translation map: 
        # Abstract function: representing the cumulative distribution function 
        # of a Zipf pmf 
        self.distMap = [x / zeta[-1] for x in zeta] 

    def next(self): 
        """Yield an integer between 0 and n, with probability governed by 
        Zipf distribution function specified by n and alpha.
        """
        # Take a uniform 0-1 pseudo-random value: 
        u = random.random()  

        # Translate the Zipf variable: 
        return bisect.bisect(self.distMap, u) - 1
    
    def __get_alpha(self):
        ans = self.alpha
        return ans
    
    def __get_n(self):
        ans = self.n
        return ans

class GaussianGenerator: 
    """
    GaussianGenerator is an immutable type representing a Gaussian CDF
    with patameters sigma and n; centre located at 1.0   

    """

    
    def __init__(self, n, sigma , mu = 20.0): 
        """Ininitialize a Gaussian CDF.
         Paramerters
         n: int 
            n >= 0
         
         sigma: float 
            sigma >= 1
            
         mu: float
        """
        # Calculate non-normalized cumulative values from 1 to n: 
        assert n >= 0 and sigma >= 1.0
        assert int(n) == n 
        self.n = n
        self.sigma = sigma
        self.mu = mu
        tmp = [math.exp(-(i - mu)**2 / (2 * sigma**2)) for i in range(1, n+1)] 
        zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0]) 

        # Store the translation map: 
        # Abstract function: representing the cumulative distribution function 
        # of a Gaussian pmf 
        self.distMap = [x / zeta[-1] for x in zeta] 

    def next(self): 
        """Yield an integer between 0 and n, with probability governed by 
        Zipf distribution function specified by n and alpha.
        """
        # Take a uniform 0-1 pseudo-random value: 
        u = random.random()  

        # Translate the Zipf variable: 
        return bisect.bisect(self.distMap, u) - 1
    
    def __get_alpha(self):
        ans = self.alpha
        return ans
    
    def __get_n(self):
        ans = self.n
        return ans

