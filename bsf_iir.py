#!/usr/bin/python

import numpy as np
import scipy as sp
import scipy.signal as sg
import matplotlib.pyplot as plt
import pylab as pl

#first trying with given values
f_num=102
del_1=0.1
del_2=0.1
f_sample=70000



if f_num > 70:
  m=f_num-70
else:
  m=f_num

if (m*0.1-int(m*0.1)) > 0:
  q_m=int(0.1*m)
else:
  q_m=(0.1*m - 1)

r_m=m-10*q_m

B_l_m=2+0.6*q_m+1.5*r_m
B_h_m=B_l_m+3

# analog values for bandstop
omega_p1=(B_l_m-1)*1000.0
omega_s1=(B_l_m)*1000.0
omega_s2=(B_h_m)*1000.0
omega_p2=(B_h_m+1)*1000.0

#analog f_range array
f_analog_a=np.array([omega_p1,omega_s1,omega_s2,omega_p2],dtype='f')

#normalized digital frequencies
f_digital_a=(f_analog_a/f_sample)*2*np.pi

#just for checking ****wrong
#f_digital_a=(f_analog_a/f_sample)

#Bilinear Transformation
f_eqv_analog_a=np.tan(f_digital_a/2)

#intermediate values needed for frequency transformation
omega_z_s=f_eqv_analog_a[0]*f_eqv_analog_a[3]
f_B=f_eqv_analog_a[3]-f_eqv_analog_a[0]


#Frequency Transformation for bandpass
f_eqv_analog_lpf_a=(f_B*f_eqv_analog_a)/(omega_z_s-(f_eqv_analog_a**2))

#Values for Butterworth lpf design
D_1=(1/(1-del_1)**2)-1
D_2=(1/del_2**2)-1
#print "D1 and D2",D_1,D_2
epsilon=np.sqrt(D_1)

mod_f_eqv_analog_lpf_a=abs(f_eqv_analog_lpf_a)
stringent_omega_s=min(mod_f_eqv_analog_lpf_a[1],mod_f_eqv_analog_lpf_a[2])
N_1=np.log(np.sqrt(D_2)/np.sqrt(D_1))/np.log(stringent_omega_s/f_eqv_analog_lpf_a[0])
N=np.ceil(N_1)

#print "stringent",stringent_omega_s
#print "omega_z_s",omega_z_s
#print "B",f_B

omega_p=f_eqv_analog_lpf_a[0]
#print "omega_p",omega_p
#Finding the filter poles
omega_c=((omega_p/(D_1**(1/(2*N))))+(stringent_omega_s/(D_2**(1/(2*N)))))/2
poles_a=np.zeros([2*N-1],dtype='complex64')
iterable=((2*k+1)*np.pi/(2*N) for k in range(2*int(N)))
xp=np.fromiter(iterable,float)

poles_a=(1.j)*omega_c*np.exp(1.j*xp)

a=1+0.j
for c in poles_a:
  if(c.real< 0):
    a=np.poly1d([1,-c],r=0)*a

#print "Low pass tf\n",a


#Lowpass filter transfer function
#Find gain for Chebyshev
K_butter=omega_c**N
#print "GAIN:",K_butter
#print "omega_c:",omega_c
numer=K_butter
denom=1+0.j
for c in poles_a:
  if(c.real<= 0):
    numer=np.poly1d([1,0,omega_z_s],r=0)*numer 
    denom=np.poly1d([-c,f_B,-c*omega_z_s],r=0)*denom

#print "analog numerator\n",numer
#print "\nanalog denominator\n",denom
z,p,k=sg.tf2zpk(numer,denom)
#print "Check",z,p,k

plt.figure(5)
plt.grid(True)
plt.scatter(p.real,p.imag,s=50,marker='x')
plt.scatter(z.real,z.imag,s=50,marker='o')
plt.title('Pole Zero plot of Analog Bandstop filter')
plt.ylabel('Imaginary')
plt.xlabel('Real')

#Tedious still basic version
#Need to try convolution
#Numerator= (omega_0^2+1)z^2+2*(omega_0^2-1)z+(omega_0^2+1) 
#Denominator= (-B-c-c*omega_0^2)z^2+(2*c-2*c*omega_0^2)z+(-c*omega_0^2-c+B)
numer=K_butter
denom=1+0.j
for c in poles_a:
  if(c.real<= 0):
    numer=np.poly1d([(omega_z_s+1),2*(omega_z_s-1),(omega_z_s+1)],r=0)*numer 
#denom=np.poly1d([(f_B-omega_z_s*c-c),(2*c-2*c*omega_z_s),(-c-f_B-c*omega_z_s)],r=0)*denom
    denom=np.poly1d([(-f_B-omega_z_s*c-c),(2*c-2*c*omega_z_s),(-c+f_B-c*omega_z_s)],r=0)*denom


#print m,q_m,r_m,B_l_m,B_h_m
#print "Analog frequencies",f_analog_a
#print "Digital frequencies",f_digital_a
#print "Equivalent Digital frequencies",f_eqv_analog_a
#print "Equivalent Analog lpf freq",f_eqv_analog_lpf_a
#print np.sqrt(D_1),np.sqrt(D_2)
#print "Order",N 
#print A_k 
#print "Poles",poles_a
print "Numerator\n",numer 
print "\nDenominator:\n",denom


#plotting poles

plt.figure(1)
plt.grid(True)
neg_poles=np.zeros([0],dtype='complex64')
for c in poles_a:
  if(c.real<= 0):
    neg_poles=np.append(neg_poles,c)
plt.scatter(neg_poles.real,neg_poles.imag,s=50,marker='x')
plt.title('Pole Zero plot of Low pass filter')
plt.ylabel('Imaginary')
plt.xlabel('Real')

nmrz=(numer.c).round(decimals=6)[::-1].real
dmrz=(denom.c).round(decimals=6)[::-1].real

z,p,k=sg.tf2zpk(nmrz,dmrz)
#print "Check",z,p,k

plt.figure(4)
plt.grid(True)
plt.scatter(p.real,p.imag,s=50,marker='x')
plt.scatter(z.real,z.imag,s=50,marker='o')
plt.title('Pole Zero plot of Digital Bandstop filter')
plt.ylabel('Imaginary')
plt.xlabel('Real')

print "\nNormalized numerator:\n",nmrz/dmrz[0]
print "\nNormalized denominator:\n",dmrz/dmrz[0]
nyq_rate=f_sample/2

#Lattice Coefficients for IIR
An=dmrz/dmrz[0]
Bn=nmrz/dmrz[0]
N=len(dmrz)
Kn=np.zeros(N)
Cn=np.zeros(N)
for i in np.arange(N-1,-1,-1):
    Kn[i]=An[i]
    Cn[i]=Bn[i]
    An_tilda=An[::-1]
    if(Kn[i] != 1):
        An=(An-(Kn[i]*An_tilda))/(1-(Kn[i]**2))
    Bn=(Bn-Cn[i]*An_tilda)
    An=np.delete(An,len(An)-1)
    Bn=np.delete(Bn,len(Bn)-1)

print "\nLattice Coefficients:Kn\n",Kn[1:]
print "\nLattice Coefficients:Cn\n",Cn

#Plot Frequency response
plt.figure(2)
plt.clf()
plt.grid(True)
w,h= sg.freqz(nmrz,dmrz,worN=512)
plt.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.title('Frequency Response')
plt.ylim(-0.05, 1.05)

# Upper inset plot.
ax1 = plt.axes([0.44, 0.56, .45, .25])
plt.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
plt.xlim(4000.0,15000.0)
plt.ylim(-0.01, 0.2)
plt.grid(True)

# Lower inset plot.
ax1 = plt.axes([0.44, 0.22, .45, .25])
plt.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
plt.xlim(4000.0,15000.0)
plt.ylim(0.96, 1.05)
plt.grid(True)

plt.figure(3)
plt.grid(True)
h_Phase = pl.unwrap(np.arctan2(np.imag(h),np.real(h)))
plt.plot(w/max(w),h_Phase)
plt.ylabel('Phase (radians)')
plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
plt.title(r'Phase response')
plt.show()
