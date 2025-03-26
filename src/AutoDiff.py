from collections import defaultdict
from typing import List
from Utils import topological_sort
from math import exp, log

class Value:
    # Constructor
    def __init__(self, value, _prevValues=(), _operator=''):
        self.value = value
        self.gradient = 0
        self._operator = _operator
        self._previous = set(_prevValues)
        self._updatePreviousGradients = lambda: None
    
    def updateGradients(self):
        # Gradient/turunan dari diri sendiri = 1
        self.gradient = 1
        update_order = topological_sort(self)
        for node in update_order:
            node._updatePreviousGradients()

    # BASIC OPERATORS
    # Operator tambah
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
            
        out = Value(self.value + other.value, (self, other), '+')

        def _updatePreviousGradients():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._updatePreviousGradients = _updatePreviousGradients

        return out

    # Operator kali
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
            
        out = Value(self.value * other.value, (self, other), '*')

        def _updatePreviousGradients():
            self.gradient += other.value * out.gradient
            other.gradient += self.value * out.gradient
        out._updatePreviousGradients = _updatePreviousGradients

        return out

    # Operator pangkat
    def __pow__(self, other):
        out = Value(self.value**other, (self,), f'**{other}')

        def _updatePreviousGradients():
            self.gradient += (other * self.value**(other-1)) * out.gradient
        out._updatePreviousGradients = _updatePreviousGradients

        return out
    
    # Negasi
    def __neg__(self):
        return self * -1
    
    # EXPONENTIAL FUNCTION
    def exp(self):
        out = Value(exp(self.value), (self,), 'exp')
        
        def _updatePreviousGradients():
            self.gradient += out.value * out.gradient
        out._updatePreviousGradients = _updatePreviousGradients

        return out
    
    # NATURAL LOGARITHM
    def log(self):
        out = Value(log(self.value), (self,), 'log')
        
        def _updatePreviousGradients():
            self.gradient += (1 / self.value) * out.gradient
        out._updatePreviousGradients = _updatePreviousGradients

        return out

    # ACTIVATION FUNCTIONS
    # RELU
    def relu(self):
        if self.value < 0:
            out = Value(0, (self,), 'relu')
        else:
            out = Value(self.value, (self,), 'relu')

        def _updatePreviousGradients():
            self.gradient += (out.value > 0) * out.gradient
        out._updatePreviousGradients = _updatePreviousGradients

        return out

    # REVERSED OPERATORS
    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(value={self.value}, gradient={self.gradient})"
    
