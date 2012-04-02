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


#Kaiser window parameters
del_omega1=f_digital_a[3]-f_digital_a[2]
del_omega2=f_digital_a[1]-f_digital_a[0]
del_omega=min(abs(del_omega1),abs(del_omega2))
A=-20*np.log10(del_1)

#Order
N_1=(A-8)/(2*2.285*del_omega)
N=np.ceil(N_1)

if(A<21):
    alpha=0
elif(A<=50):
    alpha=0.5842*(A-21)**0.4+0.07886(A-21)
else:
    alpha=0.1102(A-8.7)

omega_c1=(f_digital_a[1]+f_digital_a[0])*0.5
omega_c2=(f_digital_a[3]+f_digital_a[2])*0.5

#Ideal BPF h[n]
iterable=((np.sin(omega_c1*k)-np.sin(omega_c2*k))/(np.pi*k) for k in range(int(-N),int(N+1)))
h_ideal=np.fromiter(iterable,float)
h_ideal[N]=((omega_c1-omega_c2)/np.pi)+1
beta=alpha/N
#Generating Kaiser window 
h_kaiser=sg.kaiser(2*N+1,beta)

h_org=h_ideal*h_kaiser

print "FIR Filter Coefficients:\n",h_org

#print "Hideal",h_ideal
#print m,q_m,r_m,B_l_m,B_h_m
#print "Analog frequencies",f_analog_a
#print "Digital frequencies",f_digital_a
#print "Equivalent Digital frequencies",f_eqv_analog_a
#print "Equivalent Analog lpf freq",f_eqv_analog_lpf_a
#print np.sqrt(D_1),np.sqrt(D_2),B
#print "order=",N 
#print A_k 
#print "Poles",poles_a
#print "Numerator",numer 
#print "Denominator:",denom


#plotting poles

#plt.figure(1)
#plt.grid(True)
#plt.scatter(poles_a.real,poles_a.imag,marker='x')
#plt.legend()

nyq_rate=f_sample/2
#------------------------------------------------
# Plot the FIR filter coefficients.
#------------------------------------------------

plt.figure(1)
plt.plot(h_org, 'bo-', linewidth=2)
plt.title('Filter Coefficients (%d taps)' % (2*N+1))
plt.grid(True)

#Plot Frequency response
plt.figure(2)
plt.clf()
plt.grid(True)
w,h= sg.freqz(h_org)
plt.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.title('Frequency Response')
plt.ylim(-0.05, 1.2)

# Upper inset plot.
ax1 = plt.axes([0.42, 0.45,.45, .25])
plt.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
plt.xlim(2000.0,15000.0)
plt.ylim(0.9, 1.2)
plt.grid(True)

# Lower inset plot
ax2 = plt.axes([0.42, 0.15, .45, .25])
plt.plot((w/np.pi)*nyq_rate, np.absolute(h), linewidth=2)
plt.xlim(6000.0, 10000.0)
plt.ylim(0.0, 0.11)
plt.grid(True)

#Phase response plot
plt.figure(3)
plt.grid(True)
h_Phase = pl.unwrap(np.arctan2(np.imag(h),np.real(h)))
plt.plot(w/max(w),h_Phase)
plt.ylabel('Phase (radians)')
plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
plt.title(r'Phase response')

#Stem Diagram
plt.figure(4)
y = pl.linspace(0,61,61)
plt.stem(y,h_org,linefmt='b-', markerfmt='bo', basefmt='r-')
plt.title('Filter Coefficients (%d taps)' % (2*N+1))
plt.grid(True)

plt.show()
