# -*- coding: utf-8 -*-
# type: ignore
"""
Created on Wed Feb 15 15:42:24 2023

@author: mgrzy
"""


import numpy as np

def read_yuma(almanac_file):
    ''' 
    Reading and parsing YUMA asci format
    INPUT:
        Almanac: YUMA format 
    OUTPUT:
        almanac_data -  type list of list [strings value], number of lists is equal to number of satellite
                        one list contain satellites according to the order:         
                        ['SV ID', 'Health', 'Eccentricity', 'Time of Applicability(s)', 'Inclination(rad)', 
                        'Rate of Right Ascen(r/s)', 'SQRT(A)  (m 1/2)', 'Right Ascen at Week(rad)', 
                        'Argument of Perigee(rad)', 'Mean Anom(rad)', 'Af0(s)', 'Af1(s/s)', 'Week no']
        
    '''
    
    if almanac_file:
        alm = open(almanac_file)
        
        alm_lines = alm.readlines()
        all_sat = []
        for idx, value in enumerate(alm_lines):
            # print(idx, value)
            
            if value[0:3]=='ID:':
                one_sat_block = alm_lines[idx:idx+13]
                one_sat = []
                for line in one_sat_block:
                    data = line.split(':')
                    one_sat.append(float(data[1].strip()))
                all_sat.append(one_sat)
        alm.close()
        all_sat = np.array(all_sat)
        
        return (all_sat)


def read_alm(file):
    '''
    Parameters
    ----------
    file : .alm file
    Returns
    -------
    nav_data : 
    nav_data[0] - svprn
    nav_data[1] - health
    nav_data[2] - eccentricity
    nav_data[3] - SQRT_A (square root of major axis a [m**(1/2)])
    nav_data[4] - Omega (Longitude of ascending node at the beginning of week [deg])
    nav_data[5] - omega (Argument of perigee [deg])
    nav_data[6] - M0 (Mean anomally [deg])
    nav_data[7] - ToA (Time of Almanac [second of week])
    nav_data[8] - delta_i (offset to nominal inclination angle; i = 54 + delta_i [deg])
    nav_data[9] - Omega_dot (Rate of Right Ascension [deg/s * 1000])
    nav_data[10]- Satellite clock offset [ns]
    nav_data[11]- Satellite clock drift [ns/s]
    nav_data[12]- GPS week
   
    '''
    m = 0
    with open(file, "r") as f:
        block = []
        nav_data = []
        for s in f:
            # print(s)
            
            if m<13:
                m+=1
                block.append(s)
            else:
                block_array = np.genfromtxt(block,delimiter=10).T
                if len(np.shape(block_array))==1:
                    block_array = block_array.reshape(1,13)
                nav_data.extend(block_array)
                
                m = 0
                block = []
            
    nav_data = np.array(nav_data)        
    return nav_data

def get_prn_number(nav_data):
    prns = []
    for nav in nav_data:
        nsat = nav[0]
        if 0<nsat<=37:
            prn = int(nsat)
            prns.append(prn)
        elif 38<=nsat<=64:
            prn = 100 + int(nsat-37)
            prns.append(prn)
        elif 111<=nsat<=118:
            prn = 400 + int(nsat-110)
            prns.append(prn)
        elif 201<=nsat<=263:
            prn = 200 + int(nsat-200)
            prns.append(prn)    
        elif 264<=nsat<=310:
            prn = 300 + int(nsat-263)
            prns.append(prn)
        elif 311<=nsat:
            prn = 300 + int(nsat-328)
            prns.append(prn)           
        else: 
            prn = 500 + int(nsat)
            prns.append(prn)
    return prns

def get_alm_data(file):
    nav_data = read_alm(file)
    prns = get_prn_number(nav_data)
    nav_data[:,0] = prns
    return nav_data

def create_prn_alm(nav_data):
    prns = []
    for nav in nav_data:
        nsat = nav[0]
        if 0<nsat<=37:
            prn = 'G'+str(int(nsat)).zfill(2)
            prns.append(prn)
        elif 38<=nsat<=64:
            prn = 'R'+str(int(nsat-37)).zfill(2)
            prns.append(prn)
        elif 111<=nsat<=118:
            prn = 'Q'+str(int(nsat-110)).zfill(2)
            prns.append(prn)
        elif 201<=nsat<=263:
            prn = 'E'+str(int(nsat-200)).zfill(2)
            prns.append(prn)    
        elif 264<=nsat<=310:
            prn = 'C'+str(int(nsat-263)).zfill(2)
            prns.append(prn)
        elif 311<=nsat:
            prn = 'C'+str(int(nsat-328)).zfill(2)
            prns.append(prn)           
        else: 
            prn = 'S'+str(int(nsat)).zfill(2)
            prns.append(prn)
    return prns

def get_alm_data_str(alm_file):
    nav_data = read_alm(alm_file)
    prn = get_prn_number(nav_data)
    nav_data[:,0] = prn
    prns = create_prn_alm(nav_data) 
    return nav_data, prns    