# util packages for graph processing

from rdflib.graph import Graph
import pprint
import pandas as pd
import numpy as np

def readGraph(datalocation):
    '''Read in .nt file and print how many RDF triple the graph contains'''
    g = Graph()
    g.parse(datalocation, format="nt")
    
    print("The data contains {} RDF triples.".format(len(g)))
    
    return g

def showHeadOfGraph(graph, noObservations):
    '''Show the five first observations of the graph'''
    counter = 0

    for stmt in graph:
        pprint.pprint(stmt) 
        print()
        
        counter += 1
        
        if counter >= noObservations:
            break

def getInstanceDF(graph):
    '''Get the instances and create a pandas dataframe'''
    # instanceDF = pd.DataFrame(columns=['instance', 'class'])
    instanceDF = np.empty([len(graph), 2], dtype = "<U1000")
    counter = 0

    for stmt in graph:
        try:
            subject = stmt[0].split('/')
            subject = subject[-1]

            object = stmt[2].split('/')
            object = object[-1]

            # pprint.pprint(stmt) # for debugging

            instanceDF[counter] = [subject, object]
            # print('subject: {}, object: {}'.format(subject, object)) # for debugging
        except:
            print('Skipped: {}'.format([subject, object]))
        counter += 1
        # print(counter)
    return instanceDF

def getTypeAndSubclass(graph):
    '''Get all instances where subClassOf or type are contained in the IRI'''
    
    #counter = 0 # for debugging only take 5
    
    subClassDF = pd.DataFrame(columns = ['subClass', 'class']) # create empty dataframe for subclasses
    
    instanceDF = pd.DataFrame(columns = ['instance', 'class'])
    
    for stmt in graph:
    #   iterate over all graph triples
    
        if ('subClassOf' in stmt[1]):
            # save all subclasses into new dataframe
            subClass = stmt[0].split('/') # get subclass
            subClass = subClass[-1]
            # print('subClass: {}'.format(subClass))
            
            Class = stmt[2].split('/') # get Class
            Class = Class[-1]
            # print('Class: {}'.format(Class))
            # pprint.pprint(stmt)
            subClassDF.loc[len(subClassDF)] = [subClass, Class] # append both to dataframe
            subClassDF.append([subClass, Class])
        elif ('type' in stmt[1]):
            # save all instances into new dataframe
            instance = stmt[0].split('/')
            instance = instance[-1]
            
            Class = stmt[2].split('/')
            Class = Class[-1]
            instanceDF.loc[len(instanceDF)] = [instance, Class]            
            #counter += 1
                
        #if counter >= 5:
            #break
    
    print("Found {} subclass relations".format(len(subClassDF)))
    print("Found {} instance relations".format(len(instanceDF)))
   
    
    return subClassDF, instanceDF