import numpy as np
'''
arr = np.array([np.nan, 0, 2, np.nan, 4, 6, 8, 10])
print(arr.shape)
arr[arr<5] = -1.0
print(arr)
'''

'''
x_num, y_num, t_num, v_num = 921, 881, 19, 2

out = np.random.normal(int(x_num/2), scale=int(x_num/2), size=10)
print(out)
'''

fig = 0
print('1:', fig)

def f1():
    global fig
    fig += 1
    print('2:', fig)
    f2()
    print('3:', fig)

def f2():
    global fig
    fig += 10
    print('4:', fig)

if __name__ == '__main__':
    print('5:', fig)
    f1()
    print('6:', fig)