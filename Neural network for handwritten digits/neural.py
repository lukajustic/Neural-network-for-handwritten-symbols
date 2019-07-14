import numpy as np
import gui
import processing
np.set_printoptions(precision=3, suppress=True)

def read(path):
    lines=[]
    with  open(path) as dat:
        lines = dat.readlines()
    Xd = []
    yd = []
    for line in lines: 
        Xdata,Ydata = line.split('$')
        Ydata= Ydata.strip()
        Ydata = Ydata.split(',')
        Xdata = Xdata.split(',')
        Xdata.pop()
        Xfloat = []
        Yfloat = []
        for data in Xdata:
            Xfloat.append(float(data))
        for data in Ydata:
            Yfloat.append(float(data))
        Xd.append(Xfloat)
        yd.append(Yfloat)
    return np.asarray(Xd), np.asarray(yd)


def sigmoid(x):
    return 1/(1.+np.exp(-x))

def loss(y,yd):
    return np.mean((y-yd)**2)


def initializeWeights(ni,nh,no):
    n_i = ni
    n_h = []
    n_h.extend(nh)
    n_o = no

    Wh = []
    Wh.append(np.random.randn(n_h[0],n_i)) 
    N_hidden = len(n_h)
    for i in range(1, N_hidden): 
        Wh.append(np.random.randn(n_h[i],n_h[i-1]))
    Wo= np.random.randn(n_o,n_h[N_hidden-1])
    theta_h = []
    for n in n_h:
        theta_h.append(np.zeros((n,1)))
    theta_o = np.zeros((n_o,1))

    return {'Wh':Wh,'Wo':Wo,'th_o':theta_o,'th_h':theta_h}


def forward_pass(x, cache): 
    Wh=cache['Wh']
    Wo=cache['Wo']
    theta_o=cache['th_o']
    theta_h=cache['th_h'] 
    cache['yk'] = []
    
    #For input layer
    v = np.dot(Wh[0], x) - theta_h[0]
    yk = sigmoid(v)
    cache['yk'].append(yk)
    
    for i in range(1,len(Wh)): 
        v = np.dot(Wh[i], yk) - theta_h[i]
        yk = sigmoid(v)
        cache['yk'].append(yk)
    u=np.dot(Wo,yk)-theta_o
    y=sigmoid(u)
    
    return y

def backward_pass(x,y,yd,cache): 
    yk=cache['yk']
    n_yk = len(yk)
    Wh=cache['Wh']
    Wo=cache['Wo']
     
    EA_o = y-yd
    sigma = EA_o*y*(1-y)
    EW_o = np.dot(sigma,yk[n_yk-1].T)  #EW_o = EIo*z^T, and 'z' is the last 'yk'
    Etheta_o=-sigma
    
    if 'EW_o' not in cache:
        cache['EW_o'] = EW_o      #this is used for batch
        cache['Et_o'] = Etheta_o
        
    else:
        cache['EW_o'] += EW_o
        cache['Et_o'] += Etheta_o
    
    for i in reversed(range(0,len(Wh))): 
        if i == (len(Wh)-1):  #Take Wo from the one before the last one
            EA_h=np.dot(Wo.T,sigma)
        else:                 #For everyone else take the one before it
            EA_h=np.dot(Wh[i+1].T,sigma)
            
        sigma=EA_h*yk[i]*(1-yk[i])

        if i == 0:
            EW_h=np.dot(sigma,x.T) #If you are at the last one then x.T
        else:
            EW_h=np.dot(sigma,yk[i-1].T) #Else take the output of the previous one
        Etheta_h=-sigma
        
        if 'EW_h' not in cache:
            cache['EW_h'] = []
            cache['Et_h'] = []
            
        if len(cache['EW_h']) < len(Wh): 
            cache['EW_h'].insert(0,EW_h)
            cache['Et_h'].insert(0,Etheta_h)
        else:
            cache['EW_h'][i]+=EW_h
            cache['Et_h'][i]+=Etheta_h

def update_weights(cache,learning_rate=0.01): 
    Wh=cache['Wh']
    Wo=cache['Wo']
    theta_o=cache['th_o']
    theta_h=cache['th_h']
    EW_o=cache['EW_o']
    EW_h=cache['EW_h']
    Etheta_o=cache['Et_o']
    Etheta_h=cache['Et_h']
    
    Wo=Wo-learning_rate*EW_o
    theta_o=theta_o-learning_rate*Etheta_o
    
    cache['Wo']=Wo
    cache['th_o']=theta_o
    
    for i in range(0, len(Wh)):
        Wh[i]=Wh[i]-learning_rate*EW_h[i]
        theta_h[i]=theta_h[i]-learning_rate*Etheta_h[i]
        cache['Wh'][i] = Wh[i]
        cache['th_h'][i]=theta_h[i]
        
    del cache['EW_o']
    del cache['EW_h']
    del cache['Et_o']
    del cache['Et_h']
            

def split_into_batches(Xd, yd, batch_size): 
    X_batches = []
    y_batches = []
    for i in range(0,len(Xd), batch_size): 
        X_batches.append(Xd[i:i+batch_size])
        y_batches.append(yd[i:i+batch_size])
    return X_batches, y_batches


def backpropagation(xd,yd, cache):
    y=forward_pass(xd,cache)
    backward_pass(xd,y,yd,cache)

def train(Xd, yd, cache, epochs=10000, learning_rate=0.01, batch_size=2):
    x_batches, y_batches = split_into_batches(Xd,yd, batch_size)
    print("X batches: ", len(x_batches))
    print("Y batches: ", len(y_batches))

    q=0
    for epoch in range(0, epochs): 
            for i in range(0, len(x_batches)): 
                for j in range(0, len(x_batches[i])):
                    backpropagation(x_batches[i][j].reshape(-1,1), y_batches[i][j].reshape(-1,1), cache)
                update_weights(cache,learning_rate)
            q+=1
            if q%1000 == 0: 
                print("Epoch: ",epoch+1,", loss: ",loss(forward_pass(Xd.T, cache).T,yd))   


    

M = int(input("M: ") or 10)
path = input("Path to dataseta: ") or 'dataset2.txt'

ni = input("Input: ")
nh = input("Hidden: ")
no = input("Output: ")
nh = list(map(int, nh.split(" ")))

if 2*M != int(ni): 
    print("2*M doesn't match number of inputs")
    exit()

Xd,yd = read(path)
cache = initializeWeights(int(ni), nh, int(no))

epochs = int(input("Epochs: ") or 10000)
lrn_rt = float(input("Learning rate: ") or 0.01)
btch_sz = int(input("Batch size: ") or 1)
train(Xd, yd, cache, epochs, lrn_rt, btch_sz) 

#run forward pass for training data
print(yd)
print(forward_pass(Xd.T, cache).T)

print("=======NOW PLAY WITH IT========")


points = gui.run()
Xtest,ytest = processing.dataProcessing(points, M)
y = forward_pass(Xtest.T, cache).T

for i in range(0, len(ytest)):
    print("Y-label: ", ytest[i])
    print("Y predicted: ", y[i])
    print("Loss: ", loss(y[i], ytest[i]))
    print()
    
