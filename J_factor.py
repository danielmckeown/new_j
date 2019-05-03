import h5py
import numpy as np
import scipy.stats as st
from time import time
######################                                                                                                                                                                              


def get_data(filename, num_of_file, key1, key2):
    if num_of_file == 1:
        f = h5py.File(filename+'.hdf5', 'r')
        if key1 == 'Header':
            return f[key1].attrs[key2]
        else:
            return f[key1][key2][:]
    else:
        for i in range(0, num_of_file):
            f = h5py.File(filename+'.'+str(i)+'.hdf5', 'r')
            if key1 == 'Header':
                return f[key1].attrs[key2]
            else:
                if (len(f[key1][key2][:].shape) == 1):
                    if i == 0:
                        result = f[key1][key2][:]
                    else:
                        result = np.hstack((result, f[key1][key2][:]))
                else:
                    if i == 0:
                        result = f[key1][key2][:]
                    else:
                        result = np.vstack((result, f[key1][key2][:]))
        return result
# ============================ SWITCHES ============================ #                                                                                                                              


# ...Elvis (0) or LATTE (1)                                                                                                                                                                         
sim = 1

# ...Choose particle type (0=gas, 1=DM, 4=stars, 5=BH):                                                                                                                                             
itype = 1

# ...Snapshot                                                                                                                                                                                       
isnap = 600

# ... What run?                                                                                                                                                                                     
irun = 'dark'

# ...Number of bins in histogram                                                                                                                                                                    
my_bins = 150

# ...Out to radius (kpc) (KEEP AS FLOAT)                                                                                                                                                            
max_rad = 200.

# ...Number of bins in J-Factor grid                            



n_bins = 10

# =========================== BOOKKEEPING =========================== #                                                                                                                             

part_list = ['PartType0', 'PartType1', 'PartType2', 'PartType4']
p_type = part_list[itype]

type_list = ['Gas', 'DM', 'Disk', 'Bulge', 'Stars', 'BH']
type_tag = type_list[itype]

if sim == 0:
    base = '/data11/home/dmckeown/output/m12i/ELVIS/'
    base1 = '/data11/home/dmckeown/output/m12i/ELVIS/'
if sim == 1:
    base = '/data18/brunello/dmckeown/output'
    base1 = '/data18/brunello/dmckeown/plots/'
fname = base + '/' + irun + '/snapshot_' + str(isnap)
fname1 = base + '/' + irun + '/halo_' + str(isnap)

# ============================ LOAD DATA ============================ #                                                                                                                             


def Diff(li1, li2):
    return list(set(li1) - set(li2))


f = h5py.File(fname1 + '.hdf5', 'r')


J = np.zeros((40, 40))

h = get_data(fname, 1, 'Header', 'HubbleParam')
dm_xyz = get_data(fname, 1, p_type, 'Coordinates')/h
velocities = get_data(fname, 1, p_type, 'Velocities')

dm_mass = get_data(fname, 1, p_type, 'Masses')[0]*(1e10)/h

f = h5py.File(fname1 + '.hdf5', 'r')
halo_pos = np.array(f['position'])  # /h#/(10**3)                                                                                                                                                   

virial = np.array(f['mass.vir'])
host_velocity = np.array(f['host.velocity'])

# import operator                                                                                                                                                                                   
# index, value = max(enumerate(virial), key=operator.itemgetter(1))                                                                                                                                 
index = np.argmax(virial)

virial_radius = np.array(f['radius'])
vir = virial_radius[index]
host_v = host_velocity[index]
host_cent = halo_pos[index]
dm_xyz = dm_xyz - host_cent
radius = np.sqrt(dm_xyz[:, 0]*dm_xyz[:, 0]
                 + dm_xyz[:, 1]*dm_xyz[:, 1]
                 + dm_xyz[:, 2]*dm_xyz[:, 2])

trim_part = (radius < max_rad)
dm_xyz = dm_xyz[trim_part]
radius = radius[trim_part]

for n in range(40):
    for m in range(40):
        start = time()
        R = [0, 0, -8]
        P = [-9.5 + (n*1.0), -9.5 + (m*1.0), max_rad]
        Q = [-9.5 + (n*1.0), -10.5 + (m*1.0), max_rad]
        S = [-10.5 + (n*1.0), -9.5 + (m*1.0), max_rad]
        T = [-10.5 + (n*1.0), -10.5 + (m*1.0), max_rad]
        Center = [(n*1.0), (m*1.0), max_rad]

        RP = [a_i - b_i for a_i, b_i in zip(P, R)]
        RQ = [a_i - b_i for a_i, b_i in zip(Q, R)]
        RS = [a_i - b_i for a_i, b_i in zip(S, R)]
        RT = [a_i - b_i for a_i, b_i in zip(T, R)]
        QP = [a_i - b_i for a_i, b_i in zip(P, Q)]
        SP = [a_i - b_i for a_i, b_i in zip(P, S)]

        TQ_plane = np.cross(RT, RQ)
        QP_plane = np.cross(RQ, RP)
        SP_plane = np.cross(RP, RS)
        ST_plane = np.cross(RS, RT)
        PQS_plane = np.cross(QP, SP)

        equation1 = np.dot(TQ_plane, R)
        plane_equation1 = np.dot(dm_xyz, TQ_plane)

        equation2 = np.dot(QP_plane, R)
        plane_equation2 = np.dot(dm_xyz, QP_plane)

        equation3 = np.dot(SP_plane, R)
        plane_equation3 = np.dot(dm_xyz, SP_plane)

        equation4 = np.dot(ST_plane, R)
        plane_equation4 = np.dot(dm_xyz, ST_plane)

        index2 = ((radius < max_rad)
                  & (dm_xyz[:, 2] > -8)
                  & (plane_equation1 > equation1)
                  & (plane_equation2 > equation2)
                  & (plane_equation3 > equation3)
                  & (plane_equation4 > equation4))

        xyz = dm_xyz[index2]

        part_per_bin = np.zeros(n_bins)

        n_bins_frac = np.linspace(1.0/n_bins, 1.0, n_bins)
        # ... Leave this as is. It only sets the percentage of                                                                                                                                      
        # ... particles within each bin given your number of bins                                                                                                                                   

        # ... Binning Function                                                                                                                                                                      
        edges = st.mstats.mquantiles(xyz[:, 2], n_bins_frac)
        # ... Where z is the distance from the cone vertex                                                                                                                                          
        # ... of all the particles within the cone                                                                                                                                                  

        # ... For loop that separates all particles in each bin                                                                                                                                     

        theta = np.arctan(0.5/208.0)

        # for j in range(len(edges)):                                                                                                                                                               
        #     if j == 0:                                                                                                                                                                            
        #         mask = (xyz[:, 2] > -8.0) & (xyz[:, 2] < edges[j])                                                                                                                                
        #     else:                                                                                                                                                                                 
        #         mask = (xyz[:, 2] > edges[j - 1]) & (xyz[:, 2] < edges[j])                                                                                                                        
        # # ...This array now tells you the number of particles in each bin                                                                                                                         
        #     part_per_bin[j] = mask.sum()                                                                                                                                                          
        #     dl.insert(j, edges[j])                                                                                                                                                                
        tot_per_bin = np.round(n_bins_frac * xyz.shape[0])
        part_per_bin = np.diff(np.append(0, tot_per_bin))
        dl = np.append(-8, edges)


        # height1 = [j-i for i, j in zip(dl[:-1], dl[1:])]                                                                                                                                          
        # height = [x * 3.086e21 for x in height1]                                                                                                                                                  
        # lengths = [sum(height[:i+1]) for i in range(len(height))]                                                                                                                                 
        # base = []                                                                                                                                                                                 
        # for j in range(len(edges)):                                                                                                                                                               
        #     bases = 2.0 * lengths[j] * np.tan(theta)                                                                                                                                              
        #     base.insert(j, bases)                                                                                                                                                                 
        height = np.diff(dl) * 3.086e21
        base = np.cumsum(height) * 2.0 * np.tan(theta)

        # j_factor = np.zeros(n_bins)                                                                                                                                                               
        # for j in range(len(edges)):                                                                                                                                                               
        #     if j == 0:                                                                                                                                                                            
        #         volume = (height[j] * (base[j] * base[j])) / 3.                                                                                                                                   
        #     else:                                                                                                                                                                                 
        #         volume = (height[j] * ((base[j] * base[j])                                                                                                                                        
        #                                + (base[j - 1] * base[j])                                                                                                                                  
        #                                + (base[j - 1] * base[j - 1])                                                                                                                              
        #                                )                                                                                                                                                          
        #                   ) / 3.                                                                                                                                                                  
        pyr_vols = np.cumsum(height) * np.square(base) / 3.
        slice_vols = np.diff(np.append(0, pyr_vols))
        mass = dm_mass * part_per_bin * (5.609588e26) * (1.998e30)
        density = mass / slice_vols
        density_squared = np.square(density)
        j_factor = density_squared * height

        J[n][m] = np.sum(j_factor)
        print("Iteration took: {} seconds".format(time() - start))
        print ("J factor is" )
        print J[n][m]
J = J.ravel()
print(J)
print( J.tolist())
np.savetxt('jfactor.txt', J)



