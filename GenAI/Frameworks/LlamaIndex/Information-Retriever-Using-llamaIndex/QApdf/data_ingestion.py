from llama_index.core import SimpleDirectoryReader
import sys
from exception import customexception
from logger import logging

def load_data1():
    try:
        loader = SimpleDirectoryReader("data")
        loaded_data = loader.load_data()
       #print(loaded_data)
        return loaded_data
    except Exception as e:
        logging.info('exception in loading data')
        #raise customexception("Exception occured while loading data: "+str(e)) 
        raise customexception(e,sys)
    
def load_data():
   # print("data",data)
    try:
        loader = SimpleDirectoryReader("data")

        loaded_data = loader.load_data()
       #print(loaded_data)
        return loaded_data
    except Exception as e:
        logging.info('exception in loading data')
        #raise customexception("Exception occured while loading data: "+str(e)) 
        raise customexception(e,sys)