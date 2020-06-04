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
* 3.1_generate_transitive_subclasses.py
* 3.2_append_transitive_subclasses.py
* 3.3_remove_transitive_duplicates.py

## WebIsALOD

Data Analysis:
* 0.12_Analysis WebIsALOD.ipynb

## Matches CaLiGraph <-> WebIsALOD

Data Analysis:
* 0.8_Correlation_Matrix.ipynb
* 0.9_Topics_in_original_dataset.ipynb
* 0.10_PIDs.ipynb
* 5.1_Analysis_pids.ipynb
* 7.0_Stack_Data_and_analyze_one_hot_encoded_pids.ipynb
* 7.1_Analyze_count_pid.ipynb

Data Preparation: 
* 2.1_check_isADB_subclass.py
* 2.2_check_isADB_transitive-types.py
* 2.3_check_isADB_type.py
* 2.4_stack_matches.py
* 2.5_check_Duplicate_matches.py
* 3.4_check_isADB_transitive-subclasses.py
* 3.5_stack_transitive_subclasses.py
* 3.6_append_pid_transitive-subclasses.py
* 4.1_get_pids.py
* 4.2_append_pid_subclass.py
* 4.3_append_pid_type.py
* 4.4_append_pid_transitive-type.py
* 6.1_Generate_negative_examples.py
* 6.2_Pids_negative_examples.py
* 6.3_Stack_negative_examples.py
* 6.4_Check_pid_negative_examples.ipynb

Machine Learning:
* 8.0_ML_Preprocessing.ipynb
* 8.1_ML_One_Hot_CV_Downsampled.ipynb
* 8.2_ML_One_Hot_CV_Upsampled.ipynb
* 8.3_ML_Count_CV_Downsampled.ipynb
* 8.4_ML_Count_Upsampled_CV.ipynb
* 9.0_ML_Two_Class_Problem_Preparation.ipynb
* 9.1_ML_One_Hot_CV_Downsampled_two_class.ipynb
