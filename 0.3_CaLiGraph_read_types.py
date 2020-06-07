from helper import utilgraph
import os
import pandas as pd

folderGraph =  '/path/to/caligraph-instances_types.nt' # see http://caligraph.org/resources.html for file

files = os.listdir(folderGraph)

# for types, rdf:type is important
# the big graph has been chopped into smaller pieces in Linux since the file was very big
# chop file with: split -l numberoflines filename
### remove faulty lines such as Takashima with awk '!/Takashima/' filename > temp && mv temp filename
counter_overall = 0
counter_file = 0

for file in files:
    if '.nt' not in file:

        graph = utilgraph.readGraph(folderGraph + '/' + file)  # read graph

        utilgraph.showHeadOfGraph(graph, 5)  # show head

        classes = []
        instances = []

        for stmt in graph:
            if 'type' in stmt[1]:
                instance = stmt[0].split('/')
                instance = instance[-1]

                classe = stmt[2].split('/')
                classe = classe[-1]

                classes.append(classe)
                instances.append(instance)

                counter_overall += 1

        data = pd.DataFrame({'instance': instances, 'class': classes}, columns=['instance', 'class'])
        data.to_csv(folderGraph + '/types_' + str(counter_file) + '.csv', sep=";")
        counter_file += 1

print('Found {} type - instance relations in total'.format(counter_overall))
