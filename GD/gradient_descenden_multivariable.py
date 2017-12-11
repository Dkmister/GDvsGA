"""
Code based on website:http://firsttimeprogrammer.blogspot.com.br/2014/09/multivariable-gradient-descent.html

function used for test: six-hump camel
"""
import time
from sympy import *

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

f = (4 - 2.1*x**2 + (x**4)/3)*(x**2) + x*y + (-4+ 4 * (y ** 2))* y**2

fdx = f.diff(x)

fdy = f.diff(y)

grad = [fdx,fdy]

thetax = 1
thetay = 1
alpha =.01
iterations = 0
check = 0
precision = 10**(-6)
printData = True
maxIterations = 1000
start = time.time()
while True:
    tempthetax = thetax - alpha * N(fdx.subs(x,thetax).subs(y,thetay)).evalf()
    tempthetay = thetay - alpha * N(fdy.subs(y,thetay).subs(x,thetax)).evalf()

    iterations+=1
    if iterations > maxIterations:
        print("Too many iterations")
        printData = False
        break

    if abs(tempthetax-thetax) < precision and abs(tempthetay-thetay) < precision:
        break

    thetax = tempthetax
    thetay = tempthetay

end = time.time()

total_time = end-start

if printData:
    print("The function "+str(f)+" converges to a minimum")
    print("Number of iterations:",iterations,sep=" ")
    print("theta (x0) =",tempthetax,sep=" ")
    print("theta1 (y0) =",tempthetay,sep=" ")
    modified = f.subs({x: tempthetax, y: tempthetay})
    print("f(x) = "+str(modified))
    print(total_time)
