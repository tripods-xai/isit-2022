import numpy as np
import csv

# Use this one           
def getConstr(k,M,n,x,u,y,g,z,w,model,num_smpl,shift_length):
    
    H = np.array([[1, 1, 1, -2], [1,-1,-1,0], [-1, 1, -1,0], [-1, -1, 1, 0]])

    print('-------------- Adding constrains -----------')

    for idx_smpl in range(num_smpl):
        model.addConstrs( x[idx_smpl,idx-shift_length] + y[idx_smpl,idx] + z[idx_smpl,idx-shift_length] - 2 <= 0 for idx in range(shift_length,k))
        model.addConstrs( x[idx_smpl,idx-shift_length] - y[idx_smpl,idx] - z[idx_smpl,idx-shift_length]     <= 0 for idx in range(shift_length,k))
        model.addConstrs(-x[idx_smpl,idx-shift_length] + y[idx_smpl,idx] - z[idx_smpl,idx-shift_length]     <= 0 for idx in range(shift_length,k))
        model.addConstrs(-x[idx_smpl,idx-shift_length] - y[idx_smpl,idx] + z[idx_smpl,idx-shift_length]     <= 0 for idx in range(shift_length,k))

        # y[0]
        model.addConstrs(np.dot(H[idx][:], [ u[idx_smpl,0]*g[0] ,g[M] ,y[idx_smpl,0], 1]) <= 0 for idx in range(4))

        # y[k]
        idx_w = 1
        for idx_k in range(1,k):
            idx_u = idx_k
            idx_g = 0
            model.addConstrs(np.dot(H[idx][:], [ u[idx_smpl,idx_u]*g[idx_g] ,u[idx_smpl,idx_u-1]*g[idx_g+1] ,w[idx_smpl,idx_w], 1]) <= 0 for idx in range(4))
            idx_u = idx_u - 2
            idx_g = idx_g + 2
            for idx_g in range(2,M):
                if idx_u >= 0:
                    model.addConstrs(np.dot(H[idx][:], [ w[idx_smpl,idx_w] ,u[idx_smpl,idx_u]*g[idx_g] ,w[idx_smpl,idx_w+1], 1]) <= 0 for idx in range(4))
                    idx_u = idx_u - 1
                    idx_w = idx_w + 1
            model.addConstrs(np.dot(H[idx][:], [ w[idx_smpl,idx_w] ,g[M] ,y[idx_smpl,idx_k], 1]) <= 0 for idx in range(4))
            idx_w = idx_w + 1

def getTurboAEData(blockIdx, num_smpl, shift_length):

    def readTurboAEDate():

        with open('uData.csv',  newline='') as csvfile:
            u = [list(map(int,map(float,rec))) for rec in csv.reader(csvfile, delimiter=',')]
            u = np.array(u)
        with open('x1.csv', newline='') as csvfile:
            x1 = [list(map(int,map(float,rec))) for rec in csv.reader(csvfile, delimiter=',')]
            x1 = np.array(x1)
            for idx in range(len(x1)):
                x1[idx] = np.add(x1[idx],1); x1[idx] = np.divide(x1[idx],2)
        with open('x2.csv', newline='') as csvfile:
            x2 = [list(map(int,map(float,rec))) for rec in csv.reader(csvfile, delimiter=',')]
            x2 = np.array(x2)
            for idx in range(len(x2)):
                x2[idx] = np.add(x2[idx],1); x2[idx] = np.divide(x2[idx],2)
        with open('x3.csv', newline='') as csvfile:
            x3 = [list(map(int,map(float,rec))) for rec in csv.reader(csvfile, delimiter=',')]
            x3 = np.array(x3)
            for idx in range(len(x3)):
                x3[idx] = np.add(x3[idx],1); x3[idx] = np.divide(x3[idx],2)

        return u,x1,x2,x3
    
    u0,x1,x2,x3 = readTurboAEDate()

    uData = []
    if blockIdx == 1:
        u0 = u0[:num_smpl]
        ## Do not consider zero padding -> uData have 100 bits
        for idx in range(num_smpl): uData.append(np.array(u0[idx]).tolist())
        uData = np.array(uData)
        xData = x1[:num_smpl]
    elif blockIdx == 2:
        u0 = u0[:num_smpl]
        for idx in range(num_smpl): uData.append(np.array(u0[idx]).tolist())
        uData = np.array(uData)
        # k = len(uData[0]); n = k
        xData = x2[:num_smpl]
    elif blockIdx == 3:
    # If we are looking at block 3 (x3), we need to use the interleaver for input u
        interleaverData = [26, 86, 2, 55, 75, 93, 16, 73, 54, 95, 53, 92, 78, 13, 7, 30, 22, 24, 33, 8, 43, 62, 3, 71, 45, 48, 6, 99, 82, 76, 60, 80, 90, 68, 51, 27, 18, 56, 63, 74, 1, 61, 42, 41, 4, 15, 17, 40, 38, 5, 91, 59, 0, 34, 28, 50, 11, 35, 23, 52, 10, 31, 66, 57, 79, 85, 32, 84, 14, 89, 19, 29, 49, 97, 98, 69, 20, 94, 72, 77, 25, 37, 81, 46, 39, 65, 58, 12, 88, 70, 87, 36, 21, 83, 9, 96, 67, 64, 47, 44]
        u0 = interleaver(interleaverData,u0)[:num_smpl]
        for idx in range(num_smpl): uData.append(np.array(u0[idx]).tolist())
        uData = np.array(uData)
        xData = x3[:num_smpl]    

    return uData, xData

def interleaver(interleaver, dataList):
    numOfSample = len(dataList)
    blockLength = len(dataList[0])
    dataListNew = np.zeros((numOfSample,blockLength),int)
    for idx in range(len(interleaver)):
        loc = interleaver[idx]
        for idx_smpl in range(numOfSample):
            dataListNew[idx_smpl][idx] = dataList[idx_smpl][loc]
    return dataListNew

