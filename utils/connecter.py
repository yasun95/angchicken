import numpy as np
import smbus as sm
import time

bus = sm.SMBus(1)
address = 0x04

#SET I2C
def writeNumber(value):
    bus.write_byte(address,value)
    return -1

def readNumber():
    number = bus.read_byte(address,1)
    return number

def sendData(contents):

    x = contents[0] 
    #y = contents[1]

    # R
    if 110 < x < 140:
        x = 1
    # L
    elif 180 < x < 210:
        x = 2
    # R2
    elif 10 < x < 110:
        x = 3
    # L2
    elif 210 < x < 310:
        x = 4
    else:
        if x==0:
            x = 6
            time.sleep(4)
        else:   
            x = 5
               
    print(x)     
    data = str(x)
    data_list = list(data)

    for i in data_list:
        writeNumber(int(ord(i)))
        time.sleep(.1)

    writeNumber(int(ord(';')))
