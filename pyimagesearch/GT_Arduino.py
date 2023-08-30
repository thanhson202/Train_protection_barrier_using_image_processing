import serial
import time

DataSerial = serial.Serial('com6', 115200)
time.sleep(2)

def mo():
    DataSerial.write('mo\r'.encode())

def dong():
    DataSerial.write('dong\r'.encode())

mo()

