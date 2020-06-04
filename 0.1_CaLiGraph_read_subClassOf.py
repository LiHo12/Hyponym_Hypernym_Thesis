import pandas as pd

from helper import utilgraph

folderString = 'path/to/caligraph-ontology.nt' # see http://caligraph.org/resources.html for path

# for subClasses, the rdfs:subClassOf notation is important

graph = utilgraph.readGraph(folderString+'caligraph-ontology.nt') # read graph

print('Show head of Graph')
utilgraph.showHeadOfGraph(graph, 5) # show head

counter = 0 # count how often subclasses are found
subClasses = []
classes = []

for stmt in graph:
    # iterate over all graph triples and find subClassOf & subject + object caligraph

    if 'subClassOf' in stmt[1]:
        if ('caligraph' in stmt[0]) and ('caligraph' in stmt[2]):
            subClass = stmt[0].split('/') # get subClass
            subClass = subClass[-1]
            subClasses.append(subClass)

            classe = stmt[2].split('/')
            classe = classe[-1]
            classes.append(classe)

            counter += 1 # export when file is too big
            # print(stmt) # for debugging
            print('Found subclass: {} | class : {}'.format(subClass, classe))
            print()

data = pd.DataFrame({'subClass': subClasses, 'class': classes}, columns=['subClass', 'class'])

data.to_csv('./data/subclass_ontology0.csv', sep =";")

print('----------------------------------------------------')
print('Found {} relations in total'.format(len(subClasses)))