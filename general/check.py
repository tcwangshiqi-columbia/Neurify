import sys

with open('test.nnet','r') as f1:
    c1 = f1.readlines()

with open('models/test.nnet','r') as f2:
    c2 = f2.readlines()

round_num = 3

wrong = 0
for i in range(len(c1)):
    if i >= 7:
        arr1 = [round(x,round_num) for x in list(map(float,c1[i].split(',')[:-1]))]
        arr2 = [round(x,round_num) for x in list(map(float,c2[i].split(',')[:-1]))]

        same = [0 if arr1[i] == arr2[i] else 1 for i in range(len(arr1))]
        if sum(same) > 0:
            print('mismatch on line',i)
            wrong += 1

if wrong == 0:
    print('match')
