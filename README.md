# DL systems HW1 
This is assignment is a part of [DL-systems course presented by CMU](https://dlsyscourse.org/). 



## What you will find in this repo


This HW is the first step for building our own DL library. In this HW we will use the NUMPY as our own backend. But in the [HW3](https://github.com/ahmedtarek1325/dlsys_hw3) we will develop our own backend implementation. 

 In this HW, we start defining the operations that can be applied to tenors;
 we draft both the forward and backward implementation for the operation then we build and use a backward automatic differentiator
 to compute the backward computation stream. 

Navigating this repo you will find an implementation for the following: 
1.  Forward and Backward computation for the following operations 
    * divide                
    * divide scalar
    *  matrix multiplication
    * summation
    * reshaping
    * negation 
    * Transpose
2. Topolgical sort
3. Implement reverse mode differentiation
4. Implement softmax loss using needle operations
5. BUilding a 2 layer NN using needle operations




## Tests
To run tests you can use the following commands in the repo directory: 
ps if you are running in jupyter add `!`  before the following commands
- `python3 -m pytest -v -k "forward"`
- `python3 -m pytest -l -v -k "backward"`
- `python3 -m pytest -k "topo_sort"`
- `python3 -m pytest -k "compute_gradient"`
- `python3 -m pytest -k "softmax_loss_ndl"`
- `!python3 -m pytest -l -k "nn_epoch_ndl"`


## my AHA moments
1. **why do we run out of memory if we run the code below in pytoch?**

    The answer is basically due to the fact that the tensor does store the operation that it came from. And the operation does store its input tensors. 

    So you are not overloading the above variable; rather, you are storing every intermediate stage that happened in the memory, as this would be the way to compute the computational graph. 
    
    To solve this, we may call detach if we are not interested in the gradient calculations.


2. **How can we intuitavely imagine the gradient calculations?**
    First of all we have to consider that sometimes we are doing derivatives
    1. scalar w.r.t vectors
    2. vector w.r.t  vectors
    3. matrix w.r.t vectors

    
    $\vdots$

    not let's say for example we have 
    vector 
    
$$x_{k} = \begin{bmatrix}
a_{1} \\
a_{2} \\
a_{3}
\end{bmatrix} \tag{1}$$
    
 
let's assume that we have 

y= sum(x) = $a_1 + a_2 + a_3 \tag{2}$

if we wanted to get $\frac{dy}{d\vec{x}}$ which is the fundamental idea in back propagation -to get the output w.r.t the input- 

then we will have 

$$\frac{dy}{d\vec{x}}$ = \begin{bmatrix}
\frac{\partial (a_1 + a_2 + a_3)}{\partial a_1} \\
\frac{\partial (a_1 + a_2 + a_3)}{\partial a_2} \\
\frac{\partial (a_1 + a_2 + a_3)}{\partial a_3}
\end{bmatrix} \tag{3}$$
    
This is equivalent to 

$$\frac{dy}{d\vec{x}} = \begin{bmatrix}
1 \\
1 \\
1
\end{bmatrix} $$

Hence, we can see that on the back-propagation stream the summation does not alter or change the value of the propagted value. In addition it will backpropgate the gradients of the nodes that only participated in getting the output of the summation layer y. 
    
       
The same goes for reshaping, let's assume that we have matrix $x_{2\times3}$ s.t
    
$$x= \begin{bmatrix}
a_1 & a_2 & a_3 \\
a_4 & a_5 & a_6
\end{bmatrix} $$ 
 
if we rehsaped it to be like $y_{6\times1}$
    
$$y= \begin{bmatrix}
a_1 \\
a_2 \\
a_3 \\
a_4 \\
a_5 \\
a_6
\end{bmatrix} $$ 
    
now if we wanted again to get derivative of the output "y" w.r.t the input "x"  

Then we are having a derivative of a vector wrt matrix. 

$\therefore \frac{\partial\vec{y} }{\partial \vec{x} } = \sum_{i=1}^6 \frac{\partial a_i}{\partial \vec{x}} $

which is a derivative of a sclar w.r.t matrix. 
evaluating it intuitavely, we will conclude  that

$$ \frac{\partial\vec{y}}{\partial\vec{x}}= \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix} $$

and we can have the same conclusion as the one we had for the summation process above which is: 

reshaping does not alter the back-propagation. In addition it will backpropgate the upstream gradients for the nodes that have only participated in getting the output of the summation layer y. 
    

    
## Want to see more of the assignments ? 
### Click here to see the rest of the assignments and my take outts
1. [HW1](https://github.com/ahmedtarek1325/dlsys_hw1)
2. [HW2](https://github.com/ahmedtarek1325/dlsys_hw2)
3. [HW3](https://github.com/ahmedtarek1325/dlsys_hw3)
4. [HW4](https://github.com/ahmedtarek1325/dlsys_hw4)

## Refrences: 
1. [Youtube channel to understand the basics of tensor calculus](https://www.youtube.com/@noone988-Ahmed-Fathi/search?query=tensor%20calculus)  
2. [Lectures 4 and 5 to understand the theory](https://dlsyscourse.org/lectures/)
