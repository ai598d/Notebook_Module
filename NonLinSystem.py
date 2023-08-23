# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:19:49 2023

@author: ai598
"""

import control as ct
import numpy as np

# Nonlinear update function
def myvehicle_updt(t,x,u,params):
    m = params.get('m',50)
    j = params.get('j',12.5)
    c = params.get('c',.8)
    
    # States
    x1= x[0]
    x2= x[1]
    x3= x[2]
    x4= x[3]
    x5= x[4]
    x6= x[5]
    
    # Inputs
    fx = u[0]
    fy = u[1]
    tz = u[2]
    
   
    
    # State equations
    dx1 = x4
    dx2 = x5 
    dx3 = x6
    dx4 = (1/m)*fx*( np.cos(x3*(np.pi/180)) - np.sin(x3*(np.pi/180) )- c*x4 ) 
    dx5 = (1/m)*fy*( np.sin(x3*(np.pi/180)) + np.cos(x3*(np.pi/180) )- c*x5 )
    dx6 = (1/j)*(tz - c*x6)
    
    return [dx1,dx2,dx3,dx4,dx5,dx6]

# Output function
def myvehicle_out(t, x, u, param):
    '''
    retrun Output y 
    '''
    return x[0:6]

# Generate the Non-linear system
nonlin_myvehicle = ct.NonlinearIOSystem(myvehicle_updt, myvehicle_out, inputs=['u1','u2','u3'], outputs=['x1','x2','x3','x4','x5','x6'], states=['x1','x2','x3','x4','x5','x6'] )


def RawSystem():
    
    return nonlin_myvehicle 


# CONTROLLER

def GainMatK():
    # find equilibrium points
    u0 =np.array([0,0,0]) # no control 
    x0 = np.array([1,2,50,.5,.1,.1])
    eqpt = ct.find_eqpt(RawSystem(),x0,u0)
    xeq=eqpt[0]

    # Linearize the model at eq. points
    lin_myvehicle = ct.linearize(RawSystem(),xeq,0)
    A=lin_myvehicle.A
    B=lin_myvehicle.B
    C=lin_myvehicle.C 
    D=lin_myvehicle.D
    
    
    # Gain Matrix
    
    # Gain matrix K
    eigval = [-2,-10,-8, -6, -9, -8.5] # desired eigen values
    K = ct.place(A,B,eigval)

    Kp = np.array([K[:,0],K[:,1],K[:,2]])
    Kp = np.transpose(Kp)

    Kv = np.array([K[:,3],K[:,4],K[:,5]])
    Kv = np.transpose(Kv)
    
    return Kp,Kv,K

Kp,Kv,K = GainMatK()

sysK  = ct.append(K)
sysKp = ct.append(Kp)
sysKv = ct.append(Kv)

sysK  = ct.LinearIOSystem(sysK,inputs=['x1','x2','x3','x4','x5','x6'],outputs=['k1','k2','k3'],states=0)  # these systems has not dynamics. Therefore 0 states
sysKp = ct.LinearIOSystem(sysKp,inputs=['rkp1','rkp2','rkp3'],outputs=['rpk1','rpk2','rpk3'],states=0)
sysKv = ct.LinearIOSystem(sysKv,inputs=['rkv1','rkv2','rkv3'],outputs=['rvk1','rvk2','rvk3'],states=0)


# Connect plant and state feedback
feedsys = ct.feedback(RawSystem(),sysK,sign=-1)

# Entire system for position control
syspos = ct.series(sysKp,feedsys)

# Entire system for velocity control
sysvel = ct.series(sysKv,feedsys)


def PosControlSystem():
    return syspos

def VelControlSystem():
    return sysvel



