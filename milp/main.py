import sndhdr
from types import MethodType
from gurobipy import *
import numpy as np
from utils import *
import argparse
import time

# python optHD_final.py -num_smpl 1 -shift_length 4 -block_idx 2 -memory_length 10 -multi_solution 0
# python optHD.py -shift_length 4 -block_idx 3 -multi_solution 1 -num_smpl 10 -memory_length 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_smpl',type=int, default=10, help ='how many data sample? (int) (Default: 10)')
    parser.add_argument('-shift_length',type=int, default=4, help ='how many shift bit? (Default: 4)')
    parser.add_argument('-block_idx',type=int, default=1, help ='Which block (Default: 1)? option: 1,2,3')

    parser.add_argument('-bit_length',type=int, default=100, help ='code bit length (Default: 100 and it always be)')
    parser.add_argument('-memory_length', type=int, default=10, help ='memory length (Default: 10)')
    parser.add_argument('-multi_solution', type=int, default=0, help = 'provide multiple solution? 1 for YES. 0 for NO (Default: 0)')

    args = parser.parse_args()

    return args

start = time.time()

args = get_args()
num_smpl = args.num_smpl
blockIdx = args.block_idx
shift_length = args.shift_length

uData,xData = getTurboAEData(blockIdx, num_smpl, shift_length)
k = len(uData[0]); n = k
bit_length = args.bit_length

M = args.memory_length
MList = [args.memory_length]
objList = []; percList = []; gList = []

# Create a MIP mode
model = Model("mip")
model.setParam('Method', -1) # Options are: -1=automatic, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent, 4=deterministic concurrent, 5=deterministic concurrent simplex.

if args.multi_solution == 1:
    model.setParam('PoolSearchMode', 2) # 0(default): found one optimal result and others are by incidential; 1: found multiple result, one optimal and other no gaurantee; 2: found multiple optimal results
    model.setParam('PoolSolutions', 5) # default is 10

# Create variables
g1 = model.addVars(M+1, vtype=GRB.BINARY, name="g1")  
w = model.addVars(num_smpl,k*M, name="w") 
z = model.addVars(num_smpl,k, name="z") 
y = model.addVars(num_smpl,k, name="y")

getConstr(k,M,n,xData,uData,y,g1,z,w,model,num_smpl,shift_length)

# Cost function
cost = 0.0
total = 0
for idx_i in range(num_smpl):
    cut_edge = 0
    for idx_j in range(k-shift_length): # (bit_length - shift_length)  bits
        cost = cost + z[idx_i,idx_j]
        total = total + 1

try: 
    model.setObjective( cost, GRB.MINIMIZE)
    model.optimize()
    gg = []
    for v in model.getVars():
        if 'g' in v.varName:
            gg.append(int(v.x)) 
    print('g = ', gg)
    print('Obj: %g' % model.objVal)

    if args.multi_solution == 1:
        print('------ Start Mulitple Solution (including the optimal one - the first one) -----')
        for sNum in range(model.SolCount):
            print('Solution idx: ', sNum)
            model.setParam("SolutionNumber", sNum)
            print('Obj: %g' % model.getAttr("PoolObjVal"))
            gg = []
            for v in model.getVars():
                # print(v.varName, v.xn)
                if 'g' in v.varName:
                    gg.append(int(v.xn))
                # model.printQuality()
            print('g = ', gg)
            if gg not in gList:
                gList.append(gg)
            obj = model.getAttr("PoolObjVal")
            objList.append(obj)
            perc = float("{:.3f}".format(obj/total))
            percList.append(perc)
            print('next solution')
    else:
        gList.append(gg)
        obj = model.objVal
        objList.append(obj)
        perc = float("{:.3f}".format(obj/total))
        percList.append(perc)
    
    print('total # of bits = ', total)

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')

print('obj. List (cost) = ', objList)
print('g List (generator) = ', gList)
print('HD(%) List= ', percList)

print(model.getParamInfo('Method'))
print(args)

end = time.time()
# total time taken
print(f"Runtime of the program is {end - start}")