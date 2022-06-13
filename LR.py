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

def RMSE(x, y):
    rmse = 0
    for i in range(0, len(x)):
        rmse = rmse + (y[i] - equation(x, y, x[i])) ** 2
        # print(x[i], y[i], rmse)
    rmse = (rmse / len(x)) ** (1/2)
    return rmse

def main():
    df = pd.read_csv("regression.csv")  
    X = np.array(df['X'])
    
    #print(len(X))
    #print(X)
    y = np.array(df['Y'])
    w, b, Trace = Train(X, y, 1000, 0.000001)
    #print(w, b)
    plt.figure(figsize = (15, 4))
    plt.scatter(X, y, color = "red")
    # plt.plot([min(X), max(X)], np.array([min(X), max(X)]) * w + b, color = "black")
    plt.plot([min(X), max(X)], np.array([min(X), max(X)]) * 1.36086938 - (-0.04641726), color = "blue")
    plt.scatter(40, equation(X, y, 40), color = "green")
    plt.show()
    
    print("W: ", w)
    print("B: ", b)
    print("RMSE: ", RMSE(X,y))
    print()
    print()

main()

start = time.time()
print(time.time() - start)