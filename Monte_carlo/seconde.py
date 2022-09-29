import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def linear(para, x): # linear function
    return para[0] + para[1] * x
   
def error (p,x,y):
    return linear(p, x) - y 

def generate_y(x, para, noise_mean=0, noise_sd=1):
    noise = np.random.normal(noise_mean, noise_sd, len(x))
    y = linear(para, x) + noise * 2
    return y

def draw(x, y, y_fitted):
    plt.figure
    plt.scatter(np.array(x), np.array(y), label = 'Original curve')
    plt.plot(x, y_fitted,'-b', label ='Fitted curve')
    plt.legend()
    plt.show()

def main():
    iter_num = 500
    beta = 1
    alpha = 2.5
    groud_truth = [beta, alpha]
    para_fit = []
    vars = []

    for i in range(iter_num):
        x = np.linspace(-10,10,30)  # generate x

        y = generate_y(x, groud_truth) # generate y
        init_para = np.array([1, 3]) # init para
        para = leastsq(error, init_para, args=(x, y)) # fit

        y_fit = linear(para[0], x)
        sigma = y - y_fit
        var = np.var(sigma) / np.sum( (x - np.mean(x))**2)
        vars.append(var)
        
        para_fit.append(para[0])

    para_fit = np.concatenate(para_fit, 0).reshape((-1, 2))
    print(f"beta & alpha mean is {np.mean(para_fit, axis=0)}")
    print(f"variance mean is {np.mean(vars)}")

    plt.subplot(121)
    plt.hist(para_fit[:, 0], bins=100)
    plt.title(r"$\alpha$")

    plt.subplot(122)
    plt.hist(para_fit[:, 1], bins=100)
    plt.title(r"$\beta$")

    plt.savefig("second.png", dpi=300)
    plt.show()


 
if __name__=='__main__':
   main()