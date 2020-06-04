# Hyponym_Hypernym_Thesis

This repository contains the code for the improvement of hyponym and hypernym extractions for webdata generated from the [WebIsALOD database](https://github.com/sven-h/webisalod) and [CaLiGraph database](http://caligraph.org/).


The code is structured as follows:

## Goldstandard 
The goldstandard is represented by handlabelled statements from WebIsALOD.

Data Analysis:
* 0.0_Calculate_Goldstandard.ipynb
* 0.11_Get_Annotator_Agreement.ipynb

Data Preparation:
* 0.4_get_gold_standard.py
* 0.5_stack_data.py
* 0.6_get_pids.py
* 0.7_get_count_and_one_hot_pids_goldstandard.ipynb


## CaLiGraph

Data Preparation:
* 0.1_CaLiGraph_read_subClassOf.py
* 0.2_CaLiGraph_read_transitive-types.py
* 1.1_CaLiGraph_sanitize_subclass.py
* 1.2_CaLiGraph_sanitize_transitive-types.py
* 1.1_CaLiGraph_sanitize_types.py

## WebIsALOD

Data Analysis:
* 0.12_Analysis WebIsALOD.ipynb

## Matches CaLiGraph <-> WebIsALOD

Data Analysis:
* 0.8_Correlation_Matrix.ipynb
* 0.9_Topics_in_original_dataset.ipynb
* 0.10_PIDs.ipynb
