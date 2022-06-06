# Linear Regression
    
   
## Basic Concept of Linear Regression Method
    
Concept: Linear Regression is a regression technique that models a linear correlation between the dependent variable Y and one or more independent variables X. Linear Regression makes regression expressions using linear prediction functions, and unknown parameters are estimated from the data.
    
- Simple Linear Regression    
  Based on one variable (X), i.e. the dimension of the variable is 1.
  
- Multiple Linear Regression    
  Based on multiple variables (Xs), i.e. multi-dimensional variables  

## Regression    
A regression model is to estimate a continuous variable y through the relationship with x, which is estimated to be the cause of y.
    
<img width="1000" alt="L007" src="https://user-images.githubusercontent.com/87653966/172049570-2b6f6e26-9fa7-4bad-82dd-43bfa94ae071.png">

y becomes a target, which becomes continuous data with real numbers. Furthermore, the x is estimated using some f() regression model. However, through this estimation, the two expressions cannot be said to be the same. Therefore, the estimation cannot ultimately be the same as the original target, and there is noise in the data itself. An error occurs due to various factors, and considering this, the above equation is modified as follows.
    
<img width="1000" alt="L008" src="https://user-images.githubusercontent.com/87653966/172049569-c08a6e0a-0274-4af6-98e8-8d44814326bf.png"> 

When modeling, we must estimate the relational expression once and then estimate the parameters after estimating the relational expression. An accurate regression model cannot be created if either is incorrectly estimated. Suppose a relational expression is linearly modeled and cannot be expressed linearly (the model relational expression is incorrectly estimated), no matter how accurately estimated the parameter weight is. In that case, it is not an accurate regression model. Therefore, there is an error accordingly. Moreover, if we have estimated the relational expression well, We can make the correct regression expression by estimating the weight parameter using it.

## Linear Regression
    

A variable that changes the value of another variable is x, and a variable that changes the value dependent on the variable x is y. At this point, the value of the variable x can change independently. In contrast, the value is determined independently by the value, so X is also called an independent variable, and y is also called a dependent variable. Linear regression models the linear relationship between one or more independent variables x and y. If there is one independent variable, it is called simple linear regression.

### Simple Linear Regression Analysis
<img width="1000" alt="L001" src="https://user-images.githubusercontent.com/87653966/172049128-9b3605a2-13de-46a5-bff1-d48ba759644c.png">
- The formula above shows the formula for simple linear regression. The value w multiplied by the independent variable x is called weight in machine learning, and the value b added separately is called bias. The equation of a straight line means the gradient and intercept of the straight line, respectively. Without w and b, y and x can only express one expression: y equals x. On the graph, we can only express one straight line. 

####    

In order to infer the relationship between x and y, we set up an expression mathematically, which in machine learning is called the Hypothesis. In H(x) below, H means Hypothesis.    

####
        
<img width="1000" alt="L003" src="https://user-images.githubusercontent.com/87653966/172049275-ee2aa948-4cc2-45e0-8d5c-f24252c6ca72.png">   

####   

In previously given data, we referred to the relationship between x and y as a hypothesis to establish an expression using w and b. Furthermore, we need to find the w and b that best represent the rules for the problem. Machine learning sets up an expression that calculates the error between the actual value and the predicted value obtained from the hypothesis to find w and b and finds the optimal w and b that minimizes the value of this expression.

####

The expression for the error between the actual value and the predicted value is called the Cost function or Loss function. A function that aims to minimize or maximize the value of a function is called an objective function. Moreover, if we want to minimize the value, it is called a cost function or a loss function.

####   

Cost functions should not simply represent errors between actual data and predictive values but should be optimized to reduce errors. The Mean Squared Error (MSE) is used primarily for regression problems.


<img width="1000" alt="L010" src="https://user-images.githubusercontent.com/87653966/172052104-64403a12-fbdd-44b2-88b2-c883f7afc9e7.png">

####

To draw a straight line that best represents the relationship between y and x is to draw a straight line that is positionally closest to all the points in the figure above. Now we define the error. The error refers to the difference between the actual value y at each x in the given data and the value of H(x) predicted by the above straight line. In the figure above, Red arrows show the magnitude of the error at each point. To find the values of w and b while reducing the error, we need to find the overall size of the error. We assume that we obtain the following values for the graph y = wx + b.


<img width="1000" alt="L013" src="https://user-images.githubusercontent.com/87653966/172053375-d77c730f-99bc-46ff-8a13-19b53b573841.png">

####

We cannot figure out the absolute magnitude of the error simply by defining it as 'error = actual value - prediction' and then adding all the errors together, there is a negative error, and there is a positive error. Therefore, we use the method of adding all the errors by squares. In other words, square the distance between all points and straight lines in the figure above and add them together. If we express it in a formula, it is as follows. n means the number of data we have.

<img width="1000" alt="L015" src="https://user-images.githubusercontent.com/87653966/172053455-9860abcc-d97b-4090-a95e-d7418b758831.png">

####    

Divided by the n number of factors in the data, the mean of the sum of squares of the errors is called Mean Squared Error (MSE). The formula is as follows.

<img width="1000" alt="L016" src="https://user-images.githubusercontent.com/87653966/172053645-ad116dfd-b3ed-468b-9180-da76d67bf583.png">

####    

The predicted value of y = wx + b and the mean square error of the actual value is E. Finding w and b, which make the mean square error the minimum value, is finding the correct straight line. Overriding the mean square error as a cost function by and is:

<img width="1000" alt="L017" src="https://user-images.githubusercontent.com/87653966/172053644-08620cd7-074a-4084-b4d0-fbd73adfdaba.png">

####   

The larger the error with all points, the larger the mean square error, and the smaller the error, the smaller the mean square error. So this average peak error, i.e., if we get w and b to make Cost (w, b) minimal, we can draw a straight line that best represents the relationship between y and x.

<img width="1000" alt="L018" src="https://user-images.githubusercontent.com/87653966/172053641-3b15a96a-9771-4f14-a78c-fe7bfeed6016.png">

####   

### Optimizer: Gradient Descent    

Many machine learning, including linear regression, and deep learning, eventually work to find the parameters w and b that minimize the cost function. The algorithm used for this is called Optimizer. Furthermore, finding the appropriate w and b through this Optimizer is called training or learning in machine learning.

####    

To understand Gradient Descent, we need to understand the relationship between cost and gradient w. w is called Weight in machine learning terms, but from the point of view of the straight line equation, it means the gradient of the straight line. The graph below shows how the error increases when the gradient w is too high or too low.

<img width="1000" alt="L028" src="https://user-images.githubusercontent.com/87653966/172056694-e2efcae5-98b8-4d92-8000-70a62d1f6adb.png">

In the figure above, the blue line shows when the gradient w is w1, and the sky-blue line shows when the gradient w is w2. A straight line corresponding to y = w1x + b1, y = w2x + b2 (w1 > w2), respectively. Arrows show the error between the actual value at each point and the predicted value of the two straight lines. These are significantly larger error values than the y = wx + b straight line used in the previous prediction. In other words, if the gradient is too large, the error between the actual and predicted values increases, and if the gradient is too small, the error between the actual and predicted values increases. The same goes for b, but the error increases if b is too large or too small.

####    

For example, we make gradient descent with the hypothesis that y = wx without bias b. Let us abbreviate the cost function's value cost(w) as cost. Accordingly, the relationship between w and cost is as follows.

<img width="1000" alt="L024" src="https://user-images.githubusercontent.com/87653966/172056257-3c7cd35c-9ad6-4787-b398-0bff13c09ca4.png">

####    

As the gradient w increases to infinity, the cost value also increases to infinity, and vice versa. As the gradient w decreases to infinity, the value of cost increases to infinity. When the cost is the smallest in the graph above, it is the bottom of the convex part. The machine has to find the w that causes the cost to have the lowest value, so we have to find the value of w at the bottom of the convex part.

####    

In the figure above, the red line shows the gradient of the tangent line on the graph for five cases where w has arbitrary values. It is noteworthy that the gradient of the tangent line gradually decreases as it progresses to the bottom convexity. Furthermore, at the bottom convexity, the gradient of the tangent line is eventually zero. On the graph, the red arrow is the horizontal point.

The point where the cost is minimized is when the gradient of the tangent becomes zero and the point where the differential value becomes zero. The idea of gradient descent is to differentiate the cost function to obtain the gradient of the tangent at the current w, change the value of w in the direction where the gradient of the tangent is low, differentiate again, and repeat the process of changing the value of w toward zero gradients of the tangent.

####    

The cost function is as follows:

<img width="1000" alt="L029" src="https://user-images.githubusercontent.com/87653966/172057125-6ff245ea-a922-4a40-b83b-6f083c695834.png">

####

<img width="1000" alt="L033" src="https://user-images.githubusercontent.com/87653966/172146476-3e2be449-46aa-4e35-9acc-48fe0509f8b5.png">


####    

Now, the expression to update w to obtain w that minimizes the cost is as follows: Repeat until the gradient of the tangent is zero.

<img width="1000" alt="L030" src="https://user-images.githubusercontent.com/87653966/172057314-df906e97-e978-4046-9e11-02768c46a2dc.png">

####    

The expression above means that the gradient of the tangent line at the current w is multiplied by a and subtracted from the current w to make it a new value. A is called the learning rate here, and first, we need to look at what it means to subtract the gradient of the tangent line from the current w without thinking about a.

####    

Therefore, "a" means the learning rate. The learning rate determines how significant the change is when we change the value and let us have a value between 0 and 1. For example, it could be 0.01. The learning rate determines how wide we will move from the perspective of viewing a point on the graph and going down the slope until the slope of the tangent is zero. Intuitively, if we increase the value of the learning rate a, we can quickly find w where the gradient of the tangent line is the minimum, but it is not.

####

<img width="1000" alt="L034" src="https://user-images.githubusercontent.com/87653966/172147204-54486f8c-435f-407c-8a4e-2192cb0aeee4.png">


####

It is also important to find the appropriate value of a because the learning rate a is too low to slow down the learning rate. So far, we have learned the principle of gradient descent, focusing only on excluding b and finding the optimal w, which in fact finds the optimal values of w and b while simultaneously performing gradient descent on w and b. This is a theory of the linear regression method.

<hr>    

## Implementing Linear Regression in Python    
### Without ML Library
    
#### Cost Function Method
    
<img width="1000" alt="L033" src="https://user-images.githubusercontent.com/87653966/172146476-3e2be449-46aa-4e35-9acc-48fe0509f8b5.png">    

    def CostFunction(X, y, w, b):
        return sum((X * w + b - y) ** 2) / 2 / len(y)

####    

#### Gradient Descent Method


    # Gradient Descent W
    def GradientW(X, y, w, b):
        return sum((w * X + b - y) * X) / len(y)

    # Gradient Descent b
    def GradientB(X, y, w, b):
        return sum(w * X + b) / len(y)

####    

#### Train Method

<img width="1000" alt="L034" src="https://user-images.githubusercontent.com/87653966/172147204-54486f8c-435f-407c-8a4e-2192cb0aeee4.png">

####    

    def Train(X, y, Iteration, LearningRate):
        w = 0
        b = 0
        Trace=[]
        for _ in range(Iteration):
            Trace.append(CostFunction(X, y, w, b))
            g_w = GradientW(X, y, w, b)
            g_b = GradientB(X, y, w, b)
            w -= LearningRate * g_w
            b -= LearningRate * g_b
        return [w, b, Trace]

####    

#### Equation Method

<img width="1000" alt="L001" src="https://user-images.githubusercontent.com/87653966/172049128-9b3605a2-13de-46a5-bff1-d48ba759644c.png">    

####    

    def equation(X, y, x1):
        w, b, Trace = Train(X, y, 1000, 0.000001)
        return w * x1 + b

####    

#### Main Method

    def main():
        df = pd.read_csv("regression.csv")  
        X = np.array(df['X'])
        y = np.array(df['Y'])
        w, b, Trace = Train(X, y, 1000, 0.000001)

        plt.figure(figsize = (15, 4))
        plt.scatter(X, y, color = "red")
        plt.plot([min(X), max(X)], np.array([min(X), max(X)]) * w + b, color = "black")
        plt.scatter(40, equation(X, y, 40), color = "green")
        plt.show()

    main()
    
#### Result

<img width="1000" alt="L035" src="https://user-images.githubusercontent.com/87653966/172153176-c1f459ac-d2e8-4993-b117-16e6f5e1ed6b.png">

#### Overall Source Code

    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd


    # Cost Function
    def CostFunction(X, y, w, b):
        return sum((w * X + b - y) ** 2) / 2 / len(y)

    # Gradient Descent W
    def GradientW(X, y, w, b):
        return sum((w * X + b - y) * X) / len(y)

    # Gradient Descent b
    def GradientB(X, y, w, b):
        return sum(w * X + b) / len(y)

    # Train
    def Train(X, y, Iteration, LearningRate):
        w = 0
        b = 0
        Trace=[]
        for _ in range(Iteration):
            Trace.append(CostFunction(X, y, w, b))
            g_w = GradientW(X, y, w, b)
            g_b = GradientB(X, y, w, b)
            w -= LearningRate * g_w
            b -= LearningRate * g_b
        return [w, b, Trace]

    def equation(X, y, x1):
        w, b, Trace = Train(X, y, 1000, 0.000001)
        return w * x1 + b

    def main():
        df = pd.read_csv("regression.csv")  
        X = np.array(df['X'])
        y = np.array(df['Y'])
        w, b, Trace = Train(X, y, 1000, 0.000001)

        plt.figure(figsize = (15, 4))
        plt.scatter(X, y, color = "red")
        plt.plot([min(X), max(X)], np.array([min(X), max(X)]) * w + b, color = "black")
        plt.scatter(40, equation(X, y, 40), color = "green")
        plt.show()

    main()

    start = time.time()
    print(time.time() - start)
