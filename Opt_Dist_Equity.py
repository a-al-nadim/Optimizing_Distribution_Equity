#Importing Built-in Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import Model, GRB, quicksum

#Getting Data File
filename = pd.ExcelFile('Data.xlsx')
dataLonDC = filename.parse('DC_lon')
dataLatDC = filename.parse('DC_lat')
dataLonVC = filename.parse('VC_lon')
dataLatVC = filename.parse('VC_lat')
dataCapVC = filename.parse('VC_U')
dataDemVC = filename.parse('VC_D')

n = len(dataLatDC) #Number of DCs
m = len(dataLatVC) #Number of VCs
I=[i for i in range(0,n)];
J=[j for j in range(0,m)];
K=[k for k in range(0,m)];
A=[(i,j) for i in I for j in J]
B=[(j,k) for j in J for k in K if j!=k]

lonVC = dataLonVC.to_numpy();
latVC = dataLatVC.to_numpy();
lonDC = dataLonDC.to_numpy();
latDC = dataLatDC.to_numpy();
U = dataCapVC.to_numpy();
Dj = dataDemVC.to_numpy();

djk = np.zeros((m, m))
cij = np.zeros((n,m))
r = 6371.01

from math import radians, sin, cos, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    lon1 = radians(float(lon1.item()))
    lat1 = radians(float(lat1.item()))
    lon2 = radians(float(lon2.item()))
    lat2 = radians(float(lat2.item()))
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))
for p in range(m):
    for q in range(m):
        if p != q:
            djk[p][q]=haversine(lonVC[p],latVC[p],lonVC[q],latVC[q])
for p1 in range(n):
    for q1 in range(m):
        cij[p1][q1]=haversine(lonDC[p1],latDC[p1],lonVC[q1],latVC[q1])

mdl=Model('Project') #Model name

#Decision Variables
w = mdl.addVars(A, vtype=GRB.CONTINUOUS) #Distribution amount from DC to VC
h = mdl.addVars(J, vtype=GRB.INTEGER)

mdl.setObjective((0.035*quicksum(cij[i,j]*w[i,j] for i,j in A)), GRB.MINIMIZE)

# If U, h, or Dj are NumPy arrays, extract scalar values using .item()

mdl.addConstrs((quicksum(w[i,j] for j in J) >= 0.75 * quicksum(U[j].item() if isinstance(U[j], np.ndarray) else U[j] for j in J)) for i in I);
mdl.addConstrs((quicksum(w[i,j] for i in I) <= (U[j].item() if isinstance(U[j], np.ndarray) else U[j])) for j in J);
mdl.addConstrs((quicksum(w[i,j] for i in I) - (h[j].item() if isinstance(h[j], np.ndarray) else h[j]) == 0) for j in J);
mdl.addConstrs(((h[j].item() if isinstance(h[j], np.ndarray) else h[j])/Dj[k].item() - (h[k].item() if isinstance(h[k], np.ndarray) else h[k])/Dj[k].item() <= 0.15 for j, k in B));
mdl.addConstrs(((h[j].item() if isinstance(h[j], np.ndarray) else h[j])/Dj[j].item() - (h[k].item() if isinstance(h[k], np.ndarray) else h[k])/Dj[k].item() >= -0.15 for j, k in B));
mdl.addConstrs(w[i,j] >= 0 for i in I for j in J);
mdl.addConstrs(h[j] >= 0 for j in J);

mdl.optimize()

sol_h={j: h[j] for j in J};

ratio = list(J)
for j in J:
    ratio[j] = (h[j].x)/(Dj[j][0]);
avg_ratio = np.mean(ratio, axis = 0);
std_dev_ratio = np.std(ratio, axis = 0);

sol_w={(i,j): w[i,j] for i,j in A}
sol_w
