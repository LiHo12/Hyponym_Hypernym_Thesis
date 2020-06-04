# Hyponym_Hypernym_Thesis

This repository contains the code for the improvement of hyponym and hypernym extractions for webdata generated from the [WebIsALOD database](https://github.com/sven-h/webisalod) and [CaLiGraph database](http://caligraph.org/).


The code is structured as follows:

## Goldstandard 
The goldstandard is represented by handlabelled statements from WebIsALOD.

Data Analysis:
* 0.0_Calculate_Goldstandard.ipynb

Data Preparation:
* 0.4_get_gold_standard.py
* 0.5_stack_data.py
* 0.6_get_pids.py


## CaLiGraph

Data Preparation:
* 0.1_CaLiGraph_read_subClassOf.py
* 0.2_CaLiGraph_read_transitive-types.py

## WebIsALOD

Data Analysis:
* 0.12_Analysis WebIsALOD.ipynb
