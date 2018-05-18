import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import rayUtil as ru
import rayPlot as rp
import time
import h5py as h5

#constants
G=6.67408*10**(-11)
#m**3 kg**-1 s**-2=(kg m**2 s**-2)() 
M_solar=1.99*10**(30)
#kg
M=10*M_solar
#kg
N=((1.0*10**(-8))*M_solar)/(3.154*10**7)
#print("N is",N)
#kg/s
h=6.626070040*10**(-27)
#erg.s
c_m=2.99792458*10**8
#m/s
d=(G*M/(c_m**(2.0)))
#Conversion factor
c=2.99792458*10**10
#cm/s
sigma=5.670367*10**(-8)
#J s**-1 m**-2 K**-4
k_b=1.38064852*10**(-16)
#erg/K
r_b=6.0
R_min=6.0
R_max=60.0

def intensityDisk(X0,U0,X1,U1,v):

	x_i0 = X0[:,0]
	x_i1 = X0[:,1]
	x_i2 = X0[:,2]
	x_i3 = X0[:,3]

	x_0 = X1[:,0]
	x_1 = X1[:,1]
	x_2 = X1[:,2]
	x_3 = X1[:,3]

	u_it = U0[:,0]
	u_ix = U0[:,1]
	u_iy = U0[:,2]
	u_iz = U0[:,3]

	u_t = U1[:,0]
	u_x = U1[:,1]
	u_y = U1[:,2]
	u_z = U1[:,3]

	#x is at the source, x_p is at the observer
	F=np.zeros(x_1.shape)
	x = x_1
	y = x_2
	z = x_3
	r = np.sqrt(x*x + y*y + z*z)
	x_p = x_i1
	y_p = x_i2
	z_p = x_i3
	r_p = np.sqrt(x_p*x_p+y_p*y_p+z_p*z_p)
	theta = np.arccos(z/r)
	phi = np.arctan2(y, x)
	T=np.zeros(x_1.shape)
	I=np.zeros(x_1.shape)
	for i in range(len(x)):
		if R_min*R_min<(x[i]*x[i]+y[i]*y[i])<R_max*R_max:
			T[i]=np.power(((3.0*G*M*N/(8.0*np.pi*sigma))*((d*r[i])**(-3.0))*(1.0-(r_b/r[i])**(0.5))),0.25)
			red_1=(doppler(x_p[i],x[i],y_p[i],y[i],z_p[i],z[i],r_p[i],r[i],u_it[i],u_t[i],u_ix[i],u_x[i],u_iy[i],u_y[i],u_iz[i],u_z[i]))
			nu=(v/red_1)
			I[i]=(2.0*(h*(nu**(3.0))/c**(2.0))/((np.exp(h*nu/(k_b*T[i]))-1.0)*(np.power(red_1,3.0))))



	return I
def doppler(x_i,x,y_i,y,z_i,z,r_i,r,u_it,u_t,u_ix,u_x,u_iy,u_y,u_iz,u_z):
	#u_* are all raised components
	u_emt=(1.0-(3.0/r))**(-0.5)
	u_emx=-y*((1.0-(3.0/r))**(-0.5))*(1.0/r)**(3.0/2.0)
	u_emy=x*((1.0-(3.0/r))**(-0.5))*(1.0/r)**(3.0/2.0)
	u_obs=(np.sqrt(1.0-(2.0/r_i)))**(-1.0)
	u_test=(np.sqrt(1.0-(2.0/r)))**(-1.0)
	g_ot=-(1.0-(2.0/r_i))
	g_tt=-(1.0-(2.0/r))
	g_xx=2.0*(x**(2.0))/(r**(3.0))+1
	g_yy=2.0*(y**(2.0))/(r**(3.0))+1
	g_zz=2.0*(z**(2.0))/(r**(3.0))+1

	bottom=u_it*u_obs
	top=u_t*u_emt+u_x*u_emx+u_y*u_emy
	shift=top/bottom

	return shift

def dopplerflatspherical(x_i,x,y_i,y,z_i,z,r_i,r,u_it,u_t,u_ix,u_x,u_iy,u_y,u_iz,u_z):
	u_em=[(1.0/(np.sqrt(1.0-(1.0/r)))),0.0,0.0,(np.sqrt(1/(r**(3.0)))/(np.sqrt(1.0-(1.0/r))))]
	u_obs=[(1.0/(np.sqrt(1.0-(1.0/r_i)))),0.0,0.0,(np.sqrt(1/(r_i**(3.0)))/(np.sqrt(1.0-(1.0/r_i))))]
	top=u_t*u_em[0]+u_z*u_em[3]
	bottom=u_it*u_obs[0]+u_iz*u_obs[3]
	shift=top/bottom

	return shift

def loadRays(filename):
	map= ru.loadMap(filename)
	t1 = map.T[:,0]
	X1 = map.X[:,0,:]
	U1 = map.U[:,0,:]
	X = map.X0
	U = map.U0


	thC = map.thetaC
	phC = map.phiC
	return X,X1,U,U1,thC,phC


def plotSpectrum(rays,intensityDisk,f_range):
	X0=rays[0]
	X1=rays[1]
	U0=rays[2]
	U1=rays[3]
	thC=rays[4]
	phC=rays[5]
	F_nu=np.zeros(len(f_range))
	theta_c=[]
	phi_c=[]
	moment=time.strftime("%Y-%b-%d__%H_%M_%S",time.localtime())
	for i in range(len(thC)):
		if thC[i] not in theta_c:
			theta_c.append(thC[i])
	for j in range(len(phC)):
		if phC[j] not in phi_c:
			phi_c.append(phC[j])
	unit_area=((theta_c[-1]-theta_c[0])/(len(theta_c)-1.0))*((phi_c[-1]-phi_c[0])/(len(phi_c)-1.0))

	for v in range(0,len(f_range)):
		I=intensityDisk(X0,U0,X1,U1,f_range[v])
		F_nu[v]=unit_area * (np.sin(thC)**2 * np.cos(phC) * I).sum()
	f = open('frequency'+moment+'.txt', 'w')
	for item in f_range:
  		f.write("%s\n" % item)
  	f.close()
	g = open('Intensity'+moment+'.txt', 'w')
	for item in F_nu:
  		g.write("%s\n" % item)
  	g.close()
	fig,ax=plt.subplots()
	ax.axis([10**15,10**18,10**(-7),10**(-3)])
	plt.xlabel('Frequency Hz')
	plt.ylabel(r'Flux erg $cm^{-2}s^{-1}Hz^{-1}$')
	ax.loglog(f_range,F_nu)
	fig.savefig('Fluxfrequency.png')


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Please input Ray HDF5 files")
	rays=loadRays(sys.argv[1])
	f_range=np.linspace(10**15,10**18,1.0*10**4,endpoint=True)
	plotSpectrum(rays,intensityDisk,f_range)

