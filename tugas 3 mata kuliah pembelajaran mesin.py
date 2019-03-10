import matplotlib.pyplot as plt
import pandas as pd
import math

idx = ['x1','x2','x3','x4','types']
df = pd.read_csv("iris.csv",names=idx)
df = df.drop(df.index[100:], axis=0)
df['fakta'] = 1
df.loc[df['types'] == 'setosa','fakta'] = 0
df = df.drop(['types'], axis = 1)

#membagi data menjadi validasi dan training sesuai aturan k-fold cross validation
validasi = []
validasi.append(pd.concat([df.loc[40:49], df.loc[90:]]))
validasi.append(pd.concat([df.loc[:9], df.loc[50:59]]))
validasi.append(pd.concat([df.loc[10:19], df.loc[60:69]]))
validasi.append(pd.concat([df.loc[20:29], df.loc[70:79]]))
validasi.append(pd.concat([df.loc[30:39], df.loc[80:89]]))

training = []
training.append(df.drop(validasi[0].index))
training.append(df.drop(validasi[1].index))
training.append(df.drop(validasi[2].index))
training.append(df.drop(validasi[3].index))
training.append(df.drop(validasi[4].index))


#mencari aktivasi
def act(row, theta, bias):
	hasil = bias
	for i in range(len(row)-1):
		hasil += theta[i] * float(row[i])
	activation = 1/(1+math.exp(-hasil))
	return activation

#mencari error, prediksi, theta yang baru dan bias yang baru untuk satu kali k
def train_theta(train, theta, alpha, bias):
    new_theta = theta
    new_bias = bias
    sum_error = 0.0
    ctr = 0
    keluaran = [0,0,0,0]
    for row in train:
        activation = act(row, new_theta, new_bias)
        prediction = 1.0 if activation >= 0.5 else 0.0
        #menghitung prediksi dan fakta yang sesuai. TRUE POSITIF dan TRUE NEGATIF
        if(prediction == row[-1]):
            ctr+=1
        for i in range(len(row)-1):
            dtheta = 2*(activation-row[-1])*(1-activation)*activation*row[i]
            new_theta[i] = new_theta[i]-alpha*dtheta
        dbias = 2*(activation-row[-1])*(1-activation)*activation
        new_bias = new_bias-alpha*dbias
        error = math.pow(activation-row[-1],2)
        sum_error += error
    keluaran[0]=sum_error/len(train)
    keluaran[1]=ctr/len(train)
    keluaran[2]=new_theta
    keluaran[3]=new_bias
    return keluaran

#function untuk melakukan validasi    
def valid(train, theta, alpha, bias):
    sum_error = 0.0
    ctr = 0
    keluaran = [0,0]
    for row in train:
        activation = act(row, theta, bias)
        prediction = 1.0 if activation >= 0.5 else 0.0
        if(prediction == row[-1]):
            ctr+=1
        error = math.pow(activation-row[-1],2)
        sum_error += error
    keluaran[0]=sum_error/len(train)
    keluaran[1]=ctr/len(train)
    return keluaran

theta = [0.47,0.47,0.47,0.47]
bias = 0.4
alpha = 0.8 #dicoba juga 0.1

#untuk menampung di tiap epoch
theta_list=[theta for i in range(5)]
bias_list=[bias for i in range(5)]
train_avgerror_list=[]
train_avgdariavgerror_list=[]
train_accuracy_list=[]
train_avgaccuracy_list=[]

validasi_avgerror_list=[]
validasi_avgdariavgerror_list=[]
validasi_accuracy_list=[]
validasi_avgaccuracy_list=[]

#jumlah epoch
epoch_n = 300

#loop untuk epoch
for n in range(epoch_n):
    sum_accuracy_train = 0.0
    sum_avgerror_train = 0.0
    sum_accuracy_validasi = 0.0
    sum_avgerror_validasi = 0.0

    #loop untuk data training
    for i in range(len(training)):
        x = train_theta(training[i].values.tolist(), theta_list[i], alpha, bias_list[i])
        train_avgerror_list.append(x[0])
        train_accuracy_list.append(x[1])
        theta_list[i]=x[2]
        bias_list[i]=x[3]
        sum_avgerror_train += x[0]
        sum_accuracy_train += x[1]
    train_avgdariavgerror_list.append(sum_avgerror_train/len(training))
    train_avgaccuracy_list.append(sum_accuracy_train/5)

    #loop untuk data validasi
    for i in range(len(validasi)):
        y = valid(validasi[i].values.tolist(), theta_list[i], alpha, bias_list[i])
        validasi_avgerror_list.append(y[0])
        validasi_accuracy_list.append(y[1])
        sum_avgerror_validasi += y[0]
        sum_accuracy_validasi += y[1]
    validasi_avgdariavgerror_list.append(sum_avgerror_validasi/len(validasi))
    validasi_avgaccuracy_list.append(sum_accuracy_validasi/5)

plt.figure('Accuracy')
plt.plot(train_avgaccuracy_list,'b-', label='train')
plt.plot(validasi_avgaccuracy_list,'r-', label='validasi')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')

plt.figure('Error')
plt.plot(train_avgdariavgerror_list,'b-', label='training')
plt.plot(validasi_avgdariavgerror_list,'r-', label='validasi')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.show()
