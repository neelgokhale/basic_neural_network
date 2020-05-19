# Basic Neural Network: Flower Identification
Basic neural network programmed in python to understand nodes, weights, gradient descent and back propagation. Designed using python (JupyterLab notebook). The data set involves petal dimensions of 2 species of flowers, colored red and blue.

## Imports and Setup

```python
%matplotlib inline
```


```python
from matplotlib import pyplot as plt
import numpy as np
```
## Displaying Prelim Dataset
This matrix defines measurements of flower data
`[petal_length, petal_width, color]`

```python

data = [[3, 1.5, 1], 
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, 0.5, 1],
        [2, 0.5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

```
## Defining Sigmoid Function and Derivative

```python
def sigmoid(x):
    return 1/(1+np.exp(-x))
def dir_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
```
## Graphing Sigmoid Function and Derivative

```python

T = np.linspace(-10, 10, 100)
Y = sigmoid(T)
plt.plot(T, Y, c='r', label="Y")
Y2 = dir_sigmoid(T)
plt.plot(T, Y2, c='b', label="dY")
plt.legend(loc="upper left")
```




    <matplotlib.legend.Legend at 0x176f3279bc8>




![png](/img/output_4_1.png)

## Scatter Plot of Data

```python
# scatter data

plt.grid()
plt.axis([0, 6, 0, 6])
for i in range(len(data)):
    point = data[i]
    color = "r"
    if point[2] == 0:
        color = "b"
    plt.scatter(point[0], point[1], c=color)
```


![png](/img/output_5_0.png)

## Cost Function and Training Loop

```python

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
learning_rate = 0.3
costs = []
for i in range(500000):
    rand_index = np.random.randint(len(data))
    point = data[rand_index]
    z = point[0] * w1 + point[1] * w2 + b
    prediction = sigmoid(z)
    target = point[2]
    cost = np.square(prediction - target)
    
    costs.append(cost)
    
    dcost_dprediction = 2 * (prediction - target)
    dprediction_dz = dir_sigmoid(z)
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1
    
    dcost_dz = dcost_dprediction *dprediction_dz
    
    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db = dcost_dz * dz_db
    
    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db
    
plt.plot(costs)
```
## Resulting Weights
The testing algorithm spits out the 2 optimized weights and buffer terms: `w1, w2, b`

```python
print("w1 = ", w1, " | w2 = ", w2, " | b = ", b)

```

    w1 =  18.206320120701324  | w2 =  8.93289796400363  | b =  -65.49513657310935
    
## Predictions

```python

for i in range(len(data)):
    point = data[i]
    print(point)
    z = point[0] * w1 + point[1] * w2 + b
    prediction = sigmoid(z)
    print("prediction: {}".format(prediction))
```

    [3, 1.5, 1]
    prediction: 0.925750295079515
    [2, 1, 0]
    prediction: 1.7747640197484038e-09
    [4, 1.5, 1]
    prediction: 0.9999999990062056
    [3, 1, 0]
    prediction: 0.1252881485127762
    [3.5, 0.5, 1]
    prediction: 0.9366380163681206
    [2, 0.5, 0]
    prediction: 2.0388556064298e-11
    [5.5, 1, 1]
    prediction: 1.0
    [1, 1, 0]
    prediction: 2.199055813355231e-17
    
## Checking with Mystery Value
Mystery value `mystery_flower` is plotted on scatter plot for reference as a black square marker.

```python

plt.grid()
plt.axis([0, 6, 0, 6])
for i in range(len(data)):
    point = data[i]
    color = "r"
    if point[2] == 0:
        color = "b"
    plt.scatter(point[0], point[1], c=color)

mystery_flower = [3.1, 1.2]
plt.scatter(mystery_flower[0], mystery_flower[1], c='black', marker='X')

print("[Red, Blue] => [1, 0]")
z = mystery_flower[0] * w1 + mystery_flower[1] * w2 + b
prediction = sigmoid(z)
print(prediction)
if (prediction > 0.5):
    print("This is red")
else:
    print("This is blue")
```

    [Red, Blue] => [1, 0]
    0.8407653031944028
    This is red
    


![png](/img/output_9_1.png)


```python

```
