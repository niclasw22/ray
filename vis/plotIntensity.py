import sys
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py as h5
import rayUtil as ru
import rayPlot as rp

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
#print("the conversion factor is", d)
c=2.99792458*10**10
#cm/s
sigma=5.670367*10**(-8)
#J s**-1 m**-2 K**-4
k_b=1.38064852*10**(-16)
#erg/K
r_b=6.0
R_min=6.0
R_max=60.0


def intensityFunc(X0,U0,X1,U1,R=60.0,R_min=6.0):

	#X0 and U0 contain the lowered covariant components

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
	for i in range(len(F)):
		if R_min*R_min<(x[i]*x[i]+y[i]*y[i])<R*R:
			F[i]=1.0/(doppler(x_p[i],x[i],y_p[i],y[i],z_p[i],z[i],r_p[i],r[i],u_it[i],u_t[i],u_ix[i],u_x[i],u_iy[i],u_y[i],u_iz[i],u_z[i])**(3.0))


	return F
def intensityFuncspherical(X0,U0,X1,U1,R=60.0,R_min=6.0):

	#X0 and U0 contain the lowered covariant components

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
	for i in range(len(F)):
		if R_min*R_min<(x[i]*x[i]+y[i]*y[i])<R*R:
			F[i]=1.0/(doppler(x_p[i],x[i],y_p[i],y[i],z_p[i],z[i],r_p[i],r[i],u_it[i],u_t[i],u_ix[i],u_x[i],u_iy[i],u_y[i],u_iz[i],u_z[i])**(3.0))


	return F

def intensityBlackbody(X0,U0,X1,U1,v):
	R=60.0
	R_min=6.0

	#X0 and U0 contain the lowered covariant components

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
			T[i]=((3.0*G*M*N/(8.0*np.pi*sigma))*((d*r[i])**(-3.0))*(1.0-(r_b/r[i])**(0.5)))**(0.25)


	for i in range(len(x)):
		if T[i]!=0.0:
			red_1=doppler(x_p[i],x[i],y_p[i],y[i],z_p[i],z[i],r_p[i],r[i],u_it[i],u_t[i],u_ix[i],u_x[i],u_iy[i],u_y[i],u_iz[i],u_z[i])
			nu=v/red_1
			I[i]=2.0*(h*(nu**(3.0))/c**(2.0))/(np.exp(h*nu/(k_b*T[i]))-1.0)
			F[i]=I[i]/(red_1**(3.0))


	return F

def doppler(x_i,x,y_i,y,z_i,z,r_i,r,u_it,u_t,u_ix,u_x,u_iy,u_y,u_iz,u_z):
	#u_* are all raised components
	u_em=[(1.0-(3.0/r))**(-0.5),-y*((1.0-(3.0/r))**(-0.5))*(1.0/r)**(3.0/2.0),x*((1.0-(3.0/r))**(-0.5))*(1.0/r)**(3.0/2.0),0.0]
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
	top=u_t*u_em[0]+u_x*u_em[1]+u_y*u_em[2]
	shift=top/bottom

	return shift

def dopplerflatspherical(x_i,x,y_i,y,z_i,z,r_i,r,u_it,u_t,u_ix,u_x,u_iy,u_y,u_iz,u_z):
	u_em=[(1.0/(np.sqrt(1.0-(1.0/r)))),0.0,0.0,(np.sqrt(1/(r**(3.0)))/(np.sqrt(1.0-(1.0/r))))]
	u_obs=[(1.0/(np.sqrt(1.0-(1.0/r_i)))),0.0,0.0,(np.sqrt(1/(r_i**(3.0)))/(np.sqrt(1.0-(1.0/r_i))))]
	top=u_t*u_em[0]+u_z*u_em[3]
	bottom=u_it*u_obs[0]+u_iz*u_obs[3]
	shift=top/bottom

	return shift





def plotMap(ax, fig, filename):
	v=np.power(10.0,17)
	map= ru.loadMap(filename)
	t1 = map.T[:,0]
	X1 = map.X[:,0,:]
	U1 = map.U[:,0,:]
	X = map.X0
	U = map.U0


	thC = map.thetaC
	phC = map.phiC

	F=intensityBlackbody(X,U,X1,U1,v)
	#F = intensity_face(X1, U1)
	#F = intensity(X1, U1)

	print(F.min(), F.max())

	cf=ax.tricontourf(phC, thC, F, 256, cmap=mpl.cm.inferno)
	cbar=fig.colorbar(cf,ax=ax)
	cbar.set_label(r'Intensity erg $cm^{-2}s^{-1}Hz^{-1}$',rotation=270,labelpad=20)

	ax.set_xlim(phC.max(), phC.min())
	ax.set_ylim(thC.max(), thC.min())
	ax.set_xlabel(r'$\phi$')
	ax.set_ylabel(r'$\theta$')
	ax.set_aspect('equal')




def plotIntensity(filename,intensityFunc):

	fig, ax = plt.subplots(1, figsize=(12,9))

	plotMap(ax, fig, filename)

	fig.tight_layout()
	fig.savefig("plot.png")
	plt.close(fig)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Please input ray HDF5 files")

	plotIntensity(sys.argv[1],intensityFunc)
	#plotMapNice(sys.argv[1])
