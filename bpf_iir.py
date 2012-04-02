#!/usr/bin/python

import numpy as np
import scipy as sp
import scipy.signal as sg
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.patches as pat

#first trying with given values
f_num=102
del_1=0.1
del_2=0.1
f_sample=90000



if f_num > 70:
  m=f_num-70
else:
  m=f_num

if (m*0.1-int(m*0.1)) > 0:
  q_m=int(0.1*m)
else:
  q_m=(0.1*m - 1)

r_m=m-10*q_m

B_l_m=2+0.7*q_m+2*r_m
B_h_m=B_l_m+5

# analog values for bandpass
omega_s1=(B_l_m-1)*1000.0
omega_p1=(B_l_m)*1000.0
omega_p2=(B_h_m)*1000.0
omega_s2=(B_h_m+1)*1000.0

#analog f_range array
f_analog_a=np.array([omega_s1,omega_p1,omega_p2,omega_s2],dtype='f')

#normalized digital frequencies
f_digital_a=(f_analog_a/f_sample)*2*np.pi

#just for checking ****wrong
#f_digital_a=(f_analog_a/f_sample)

#Bilinear Transformation
f_eqv_analog_a=np.tan(f_digital_a/2)

#intermediate values needed for frequency transformation
omega_z_s=f_eqv_analog_a[1]*f_eqv_analog_a[2]
f_B=f_eqv_analog_a[2]-f_eqv_analog_a[1]

#Frequency Transformation for bandpass
f_eqv_analog_lpf_a=((f_eqv_analog_a**2)-omega_z_s)/(f_B*f_eqv_analog_a)

#Values for Chebyschev lpf design
D_1=(1/(1-del_1)**2)-1
D_2=(1/del_2**2)-1
epsilon=np.sqrt(D_1)

mod_f_eqv_analog_lpf_a=abs(f_eqv_analog_lpf_a)
stringent_omega_s=min(mod_f_eqv_analog_lpf_a[0],mod_f_eqv_analog_lpf_a[3])
N_1=np.arccosh(np.sqrt(D_2)/np.sqrt(D_1))/np.arccosh(stringent_omega_s/f_eqv_analog_lpf_a[2])
N=np.ceil(N_1)

#print "stringent",stringent_omega_s
#print "omega_z_s",omega_z_s
#print "B",f_B

omega_p=f_eqv_analog_lpf_a[2]
#Finding the filter poles
poles_a=np.zeros([2*N-1],dtype='complex64')
iterable=((2*k+1)*np.pi/(2*N) for k in range(2*int(N)))
A_k=np.fromiter(iterable,float)
B=np.arcsinh(1/epsilon)/N
poles_real_a=omega_p*np.sin(A_k)*np.sinh(B)
poles_img_a=omega_p*np.cos(A_k)*np.cosh(B)

poles_a=poles_real_a+poles_img_a*(1.j)


#Lowpass filter transfer function
#Find gain for Chebyshev
a=1+0.j
K_cheby=1

for c in poles_a:
  if(c.real< 0):
    a=np.poly1d([1,-c],r=0)*a

#print "Low pass tf\n",a

for c in poles_a:
  if(c.real< 0):
    a=np.poly1d([1,-c],r=0)*a

for k in range(int(N)):
    K_cheby=K_cheby*poles_a[k]

#print "GAIN:",K_cheby

#Hanalog(s) s_l <--F(s)
#For BPF when the transformation is applied to 1/(s-c)
#we get Bs/(s^2+omega_0^2-B*c*s).Using this basic result and
#finding numerator and denominator separately
#For bandstop numerator and denominator will be interchanged
numer=K_cheby.real
denom=1+0.j
for c in poles_a:
  if(c.real<= 0):
    numer=np.poly1d([f_B,0],r=0)*numer 
    denom=np.poly1d([1,-f_B*c,omega_z_s],r=0)*denom

#print "analog numerator\n",numer
#print "\nanalog denominator\n",denom
z,p,k=sg.tf2zpk(numer,denom)

plt.figure(5)
plt.grid(True)
plt.scatter(p.real,p.imag,s=50,c='b',marker='x')
plt.scatter(z.real,z.imag,s=50,c='b',marker='o')
plt.title('Pole Zero plot of Analog Bandpass filter')
plt.ylabel('Imaginary')
plt.xlabel('Real')

#Tedious still basic version
#Need to try convolution
#Numerator= B(z^2-1)
#Denominator= (omega_0^2-B+1)z^2+(2*omega_0^2-2)z+(omega_0^2+B+1)
numer=K_cheby.real
denom=1+0.j
for c in poles_a:
  if(c.real<= 0):
    numer=np.poly1d([f_B,0,-f_B],r=0)*numer 
    denom=np.poly1d([(omega_z_s-f_B*c+1),((2*omega_z_s)-2),(omega_z_s+f_B*c+1)],r=0)*denom

z,p,k=sg.tf2zpk(numer,denom)
#print "Check",z,p,k

plt.figure(4)
plt.grid(True)
plt.scatter(p.real,p.imag,s=50,marker='x')
plt.scatter(z.real,z.imag,s=50,marker='o')
plt.title('Pole Zero plot of Digital Bandpass filter')
plt.ylabel('Imaginary')
plt.xlabel('Real')

#print m,q_m,r_m,B_l_m,B_h_m
#print "Analog frequencies",f_analog_a
#print "Digital frequencies",f_digital_a
#print "Equivalent Digital frequencies",f_eqv_analog_a
#print "Equivalent Analog lpf freq",f_eqv_analog_lpf_a
#print "D1,D2",D_1,D_2
#print "order=",N 
#print A_k 
#print "Poles",poles_a
print "Numerator\n",numer 
print "\nDenominator\n",denom


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

#nmrz=(numer.c).round(decimals=6)[::-1].real
#dmrz=(denom.c).round(decimals=6)[::-1].real
nmrz=(numer.c).round(decimals=6).real
dmrz=(denom.c).round(decimals=6).real


print "\nNormalized numerator Array:\n",nmrz/dmrz[0]
print "\nNormalized denominator Array:\n",dmrz/dmrz[0]
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
plt.ylim(-0.05, 1.1)

# Upper inset plot.
ax1 = plt.axes([0.44, 0.6, .45, .25])
plt.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
plt.xlim(5000.0,15000.0)
plt.ylim(0.85, 1.1)
plt.grid(True)

plt.figure(3)
plt.grid(True)
h_Phase = pl.unwrap(np.arctan2(np.imag(h),np.real(h)))
plt.plot(w/max(w),h_Phase)
plt.ylabel('Phase (radians)')
plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
plt.title(r'Phase response')

plt.show()

