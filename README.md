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
* 10.6_Extract_provids_goldstandard.py
* 10.7_Explode_provids_and_pids_goldstandard.py
* 10.8_Explode_provids_and_pids_seperately_goldstandard.py
* 10.9_merge_contexts_goldstandard.py
* 10.10_calculate_distance_goldstandard.py
* 10.11_stack_mean_sums_goldstandard.py
* 10.12_stack_sentences_goldstandard.py
* 10.13_get_contexts_positives_goldstandard.py

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
* 10.0_get_unique_values.py
* 10.1_get_modifications_two_class.py
* 10.2_Extract_provids.py
* 10.3_Explode_provids_and_pids.py
* 10.4_Explode_provids_and_pids_seperately.py
* 10.5_merge_contexts.py
* 11.0_get_index_pattern.py
* 11.1_calculate_distance.py
* 11.2_groupby_pids_mean_validation.py
* 11.3_group_ids.py
* 11.4_Pivot_counts.py
* 14.0_Calculate_Sentence_Length.py
* 14.1_Group_provids_ids.py
* 15.0_Get_test_modifications.py
* 15.1_Extract_provids_test.py
* 15.2_Explode_provids_pids_test.py
* 15.3_Explode_provids_and_pids_seperateley_test.py
* 15.4_merge_contexts_test.py
* 15.5_calculate_distance_test.py
* 15.6_stack_means_sums_test.py
* 15.7_Testing_on_test_set.ipynb

Machine Learning:
* 8.0_ML_Preprocessing.ipynb
* 8.1_ML_One_Hot_CV_Downsampled.ipynb
* 8.2_ML_One_Hot_CV_Upsampled.ipynb
* 8.3_ML_Count_CV_Downsampled.ipynb
* 8.4_ML_Count_Upsampled_CV.ipynb
* 9.0_ML_Two_Class_Problem_Preparation.ipynb
* 9.1_ML_One_Hot_CV_Downsampled_two_class.ipynb
* 9.2_ML_Count_CV_Downsampled_two_class.ipynb
* 9.3_ML_Count_CV_Downsampled_two_class_no_stratify.ipynb
* 9.4_ML_One_Hot_CV_Fraction_two_class.ipynb
* 9.5_Get_count_columns.ipynb
* 9.6_ML_V2_Count_CV_Downsampled_two_class.ipynb
* 9.8_ML_V3_Count_CV_Downsampled_two_class_with_dimensionality_reduction.ipynb
* 9.10_ML_One_Hot_CV_Fraction_two_class_without_sampling.ipynb
* 9.11_SMOTE.ipynb
* 9.12_Get_fraction_count_based.ipynb
* 9.13_Autoencoder_Approach.ipynb
* 12.0_3_class_new.ipynb
* 12.1_One_Hot_Three_class_subclass.ipynb
* 12.2_One_Hot_Three_class_types.ipynb
* 12.3_Count_Three_class_subclass.ipynb
* 12.4_Count_Three_class_types.ipynb
* 12.6_Test_Count_Naive_Bayes.ipynb
* 12.7_Confusion_Matrix_NB_Test.ipynb
* 13.0_Distance_Features_fold.ipynb
* 13.1_Distance_Cross_Validation.ipynb
* 13.2_Distance_Mean_Two_class_dowsampled.ipynb
* 13.3_Distance_Sum_Two_class_dowsampled.ipynb
* 13.4_Confusion_Matrix_NB_Test_Two_class.ipynb
* 13.5_SMOTE_Two_Class.ipynb
