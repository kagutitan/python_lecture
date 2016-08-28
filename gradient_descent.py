import numpy as np
import matplotlib.pyplot as plt
import random

def calc_value(x, param):
    value = 0
    for n, w in enumerate(param):
        value += (x ** n) * w
    return value

def calc_value_grad(x, param):
    grad = []
    for n, w in enumerate(param):
        grad.append(x ** n)
    return np.array(grad)


def calc_error_grad(sample, param):
    diffs = []
    for k, w in enumerate(param):
        diff = 0
        for s in sample:
            diff += -2.0 * (s["y"] - calc_value(s["x"], param)) * (s["x"] ** k)
        diffs.append(diff)
    return np.array(diffs)

def sgd(sample, param, mue):
    for _ in range(1000000):
        param_next = param - mue * calc_error_grad(sample, param)
        error = calc_error(sample, param_next)
        print("error: ", error)
        #print("error_grad: ", calc_error_grad(sample, param_next))
        #print(np.linalg.norm(param_next - param))
        #if np.linalg.norm(param_next - param) < 0.01:
        if error < 0.1:
            break
        param = param_next

    return param_next


def function(x):
    return 0.5 * x ** 2 - x + 5

def make_sample():
    inputs = [random.random() * 10 for _ in range(20)]
    outputs = [function(x) for x in inputs]
    sample = []
    for x, y in zip(inputs, outputs):
        sample.append({"x": x, "y": y})
    return sample

if __name__ == "__main__":
    degree = 20
    w = np.array([0] * degree)

    #sample = [{"x": 0, "y": 0}, {"x": 1, "y": 2}, {"x": 2, "y": 4}]
    sample = make_sample()
    print(sample)

    w = sgd(sample, w, 0.00001)
    print(w)

    plt.hold(True)
    for s in sample:
        plt.plot([s["x"]], [s["y"]], marker = "o")
    inputs = [0.01 * x for x in range(1000)]
    outputs = [calc_value(x, w) for x in inputs]
    plt.plot(inputs, outputs)
    plt.show()
    
def calc_error(sample, param):
    mse = 0
    for s, w in zip(sample, param):
        mse += (s["y"] - calc_value(s["x"], param)) ** 2
    return mse