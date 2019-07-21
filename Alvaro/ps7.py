#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:01:54 2019

@author: helios
"""

def search(L, e):
    for i in range(len(L)):
        if L[i] == e:
            return True
        if L[i] > e:
            return False
    return False

def newsearch(L, e):
    size = len(L)
    for i in range(size):
        if L[size-i-1] == e:
            return True
        if L[i] < e:
            return False
    return False

def swapSort(L): 
    """ L is a list on integers """
    print("Original L: ", L)
    for i in range(len(L)):
        for j in range(i+1, len(L)):
            if L[j] < L[i]:
                # the next line is a short 
                # form for swap L[i] and L[j]
                L[j], L[i] = L[i], L[j] 
                #print(L)
    print("Final L: ", L)
    
def modSwapSort(L): 
    """ L is a list on integers """
    print("Original L: ", L)
    for i in range(len(L)):
        for j in range(len(L)):
            if L[j] < L[i]:
                # the next line is a short 
                # form for swap L[i] and L[j]
                L[j], L[i] = L[i], L[j] 
                #print(L)
    print("Final L: ", L)

L = [0,1,2,3,4,5]
print(swapSort(L))

L = []
print(swapSort(L))

L = [2,3]
print(swapSort(L))

L = [3]
print(swapSort(L))

L = [3,4,5, 1,2]
print(swapSort(L))

L = []
print(swapSort(L))

L = [2,3,7,6,5]
print(swapSort(L))

L = [3]
print(swapSort(L))

##############
L = [0,1,2,3,4,5]
print(modSwapSort(L))

L = []
print(modSwapSort(L))

L = [2,3]
print(modSwapSort(L))

L = [3]
print(modSwapSort(L))

L = [3,4,5, 1,2]
print(modSwapSort(L))

L = []
print(modSwapSort(L))

L = [2,3,7,6,5]
print(modSwapSort(L))

L = [3]
print(modSwapSort(L))