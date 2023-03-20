"""Operator implementations."""

from numbers import Number
from turtle import shape
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * power_scalar(node.inputs,(self.scalar -1)/self.scalar )
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        num,den = node.inputs
        return out_grad/den , - out_grad*num / power_scalar(den,2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None: self.axes = (len(a.shape)-2,len(a.shape)-1)
        return array_api.swapaxes(a,*self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        changed_axes_ = node.op.axes
        return out_grad.transpose(changed_axes_)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        grad_shape = out_grad.shape

        # if the two shapes are not equal then what broadcasting
        # do is that it start adds ddimensions from teh left
        # hence we see if those dim are there to get them with us
        if len(input_shape) != len(grad_shape): 
            i= len(grad_shape)- len(input_shape)
            axes= list(range(i))       
        else : i,axes =0, [] 

        j= 0
        # now we want to know the dimensions that did exist form the begging
        # but they have been broadcasted into higher values 
        # hence we do compare between shapes if not equal then it means
        # a broadcasting had happen
        while j< len(input_shape): 
            if input_shape[j] != grad_shape[j+i]: axes+=[i+j]
            j+=1


        out_grad=summation(out_grad,tuple(axes))
        return reshape(out_grad,input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a,axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        # we want to know the shape of the tensor
        # before summmation
        shape_,axes= list(node.inputs[0].shape), self.axes

        if shape_ is None: 
            shape_ = [1 for i in axes]
        elif axes is not None: 
            if type(axes)==int : axes = [axes]
            shape_= [ v if i not in axes else 1 for i,v in enumerate(shape_) ] 
        else: 
            shape_ = [1 for i in  shape_]
        out_grad=reshape(out_grad,tuple(shape_)) 
        out_grad =  broadcast_to(out_grad,node.inputs[0].shape)
        return out_grad
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhm,rhm = node.inputs
        l_range,r_range = len(out_grad.shape) - len(lhm.shape) ,len(out_grad.shape) - len(rhm.shape) 
        return summation(out_grad @ rhm.transpose(), tuple(range(l_range)))\
               , summation(lhm.transpose() @ out_grad ,tuple(range(r_range)))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a+1e-7)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad,node.inputs[0]+1e-7) # 1 here represents the smoothing value
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return  out_grad* node.data
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION     
        a = a * (a>0)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        derv= node.realize_cached_data()
        return Tensor((derv>0)) * out_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

