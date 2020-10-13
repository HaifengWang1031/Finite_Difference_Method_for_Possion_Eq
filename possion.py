import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# right term
f = lambda x,y: 0

# dirichlet boundary function
omega = 10*np.pi
g = lambda x,y: np.sin(omega * y)*np.exp(-omega*x)

#domain
x_limit = (0,1)
y_limit = (0,1)

# num of partition
N = 100

delta_x = (x_limit[1] - x_limit[0])/N
delta_y = (y_limit[1] - y_limit[0])/N

# label_matrix
lm = np.zeros([(N+1)**2,7])
ip_index = -1
for i in range(len(lm)):
    x_l = i%(N+1)
    y_l = i // (N+1)
    x = x_l * delta_x
    y = y_l * delta_y
    if y in y_limit or x in x_limit:
        is_boundary = 1
        value = g(x,y)
        ip = -1
    else:
        is_boundary = 0
        value = 0
        ip_index += 1
        ip = ip_index
    lm[i] = np.array([x_l,y_l,x,y,is_boundary,ip,value])

# interior point number
ip_num = (N-1)**2
A = np.zeros((ip_num,ip_num))
b = np.zeros((ip_num,1))

# interior point to
ip_dict = {}
for i,ip in enumerate(lm[:,5]):
    if ip != -1:
        ip_dict[ip] = i


def pos2index(pos):
    for i,ipos in enumerate(lm[:,0:2]):
        if pos == list(ipos):
            return i

for i in range(ip_num):
    A[i,i] = 2/delta_x + 2/delta_y

    lm_index = ip_dict[i]
    u_p = [lm[lm_index][0],lm[lm_index][1]+1]
    u_i = pos2index(u_p)
    d_p = [lm[lm_index][0],lm[lm_index][1]-1]
    d_i = pos2index(d_p)
    l_p = [lm[lm_index][0]-1,lm[lm_index][1]]
    l_i = pos2index(l_p)
    r_p = [lm[lm_index][0]+1,lm[lm_index][1]]
    r_i = pos2index(r_p)

    b[i] = f(lm[lm_index][2],lm[lm_index][3])

    if lm[u_i,4] == 1:
        b[i] += lm[u_i,6]/delta_y
    else:
        A[i,int(lm[u_i,5])] = -1/delta_y

    if lm[d_i,4] == 1:
        b[i] += lm[d_i,6] / delta_y
    else:
        A[i, int(lm[d_i,5])] = -1 / delta_y

    if lm[l_i,4] == 1:
        b[i] += lm[l_i,6]/delta_x
    else:
        A[i, int(lm[l_i,5])] = -1/delta_x
    if lm[r_i,4] == 1:
        b[i] += lm[r_i,6]/delta_x
    else:
        A[i,int(lm[r_i,5])] = -1/delta_x


u_i = np.dot(np.linalg.inv(A),b)

for i in range(ip_num):
    lm_index = ip_dict[i]
    lm[lm_index,6] = u_i[i]

x = lm[:,2]
y = lm[:,3]
u = lm[:,6]

x = x.reshape((N+1,N+1))
y = y.reshape((N+1,N+1))
u = u.reshape((N+1,N+1))

fig,axes = plt.subplots(figsize = (8,8),subplot_kw={'projection':'3d'})

axes.plot_surface(x,y,u,cmap = mpl.cm.Blues)
plt.show()
