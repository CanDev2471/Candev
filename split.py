import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import random


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(149,50)
        self.fc2 = nn.Linear(50,12)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

f = open("factors.txt",'r')
fd = f.read()
f.close()
fd = fd.replace(";",",")
fs = fd.split("\n")
fs = fs[11:]

fl = [i.split(",") for i in fs]
l = []
for i in fl:
    for j in i:
        l.append(j)
l = [i.split(";") for i in l]

l2 = []
for i in l:
    for j in i:
        l2.append(j)

while True:
    try:
        l2.remove("")
    except:
        break

N =  len(set(l2))
s = list(set(l2))

v=[[i.count(j) for j in s] for i in fl]
v = np.array(v)#,dtype=np.half)


r = [(s[i],np.sum(v[:,i])) for i in range(len(v[0]))][::-1]
rf = [np.sum(v[:,i])/len(v[:,0]) for i in range(len(v[0]))][::-1]
#o = open("vects.txt","w")
#for i in v:
#    o.writelines(str(i)+"\n")
#o.close()

itf = open("incident_type.txt")
it = itf.read()
itf.close()
it = it.split("\n")
it2 = [i for i in it]
while True:
    try:
        it.remove("")
    except:
        break

it3 = [i.split(",") for i in it]
it = [i.split(",") for i in it]
il = []
for i in it:
    for j in i:
        il.append(j)

sit = set(il)
iv = [[i.count(j) for j in sit] for i in it]

v=v.tolist()
total_range = list(range(len(v)))
random.shuffle(total_range)
train_range = total_range[:1000]
test_range = total_range[1001:]
net = NN()
lrp = 0.00000001
m0 = 0.0
opt = opt.SGD(net.parameters(),lr=lrp,momentum=m0)
#crit = nn.CrossEntropyLoss()
crit = nn.MSELoss()

epoch_mult = 40  
batch_mult = 20
batch_size = int(len(train_range)/batch_mult)
earg = range(len(v))
for epb in range(50):
    #earg = random.shuffle(earg)
    for epoch in range(epoch_mult):
        for batch in range(batch_mult):
            bdata = torch.FloatTensor([v[i] for i in train_range[batch*batch_size:(batch+1)*batch_size]])#i*batch_size:(i+1)*batch_size])
            blab= torch.FloatTensor([iv[i] for i in train_range[batch*batch_size:(batch+1)*batch_size]])#i*batch_size:(i+1)*batch_size])
            #blab = torch.FloatTensor(iv[i*batch_size:(i+1)*batch_size])
            outs = net(bdata)
            loss = crit(outs,blab)
            loss.backward()
            opt.step()
    print(loss)
    dum = np.random.randint(0,10)
print((torch.round(outs[dum]/max(outs[dum]))).tolist(),np.argmax((torch.round(outs[dum]/max(outs[dum]))).tolist()))
print(blab[dum].tolist(),np.argmax(blab[dum].tolist()))
bdata = torch.FloatTensor([v[i] for i in test_range])
blab= torch.FloatTensor([iv[i] for i in test_range])
outs = net(bdata)

outs = (torch.round(outs/torch.max(outs))).tolist()
blab = blab.tolist()
wins = 0
for i in range(len(test_range)):
    if str(np.argwhere(blab[i]).flatten()) == str(np.argwhere(outs[i]).flatten()):
        wins+=1
print("prediction accuracy: ",wins/len(test_range))

#torch.save(net.state_dict(),"predictor_net.pt")



ins = torch.eye(len(v[0]))
pout = []
for i in ins:
    pout.append(net(i).tolist())
pout =np.array(pout)
pfg = pout[:,0]

num = [pfg[i]*rf[i] for i in range(len(pfg))]
deno = np.array(num)
deno = np.sum(deno)
num = num/deno

pef= num/deno
