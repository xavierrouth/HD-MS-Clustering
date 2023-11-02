#pragma once

#include <hpvm_hdc.h>
#include <heterocc.h>
#include <iostream>

//#define HAMMING_DIST

#undef D
#undef N_FEATURES
#undef K

typedef int binary;
typedef float hvtype;


#ifdef HAMMING_DIST
#define SCORES_TYPE hvtype
#else
#define SCORES_TYPE float
#endif

#ifndef DEVICE
#define DEVICE 1
#endif

// RANDOM PROJECTION ENCODING!!
// Matrix-vector mul
// Encodes a single vector using a random projection matrix
//
// RP encoding reduces N_features -> D 

template <typename T>
T zero_hv(size_t loop_index_var) {
	return 0;
}



// clustering_node is the hetero-c++ version of searchUnit from the original FPGA code.
// It pushes some functionality to the loop that handles the iterations.
// For example, updating the cluster centers is not done here.
// Initializing the centroids is not done here.
// Node gets run max_iterations times.
// Dimensionality, number of clusters, number of vectors

// In the streaming implementation, this runs for each encoded HV, so N_VEC * EPOCHs times.
template<int D, int K>
void __attribute__ ((noinline)) clustering_node(/* Input Buffers: 3*/
        __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, 
        __hypermatrix__<K, D, hvtype>* clusters_ptr, size_t clusters_size, 
        __hypermatrix__<K, D, hvtype>* temp_clusters_ptr, size_t temp_clusters_size, // ALSO AN OUTPUT
        __hypervector__<K, SCORES_TYPE>* scores_ptr, size_t scores_size, // Used as Local var.
        __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  // Used in second stage of clustering node for extracting and accumulating
        int encoded_hv_idx,
        /* Output Buffers: 1*/
        int* labels, size_t labels_size) { // Mapping of HVs to Clusters. int[N_VEC]

    void* section = __hetero_section_begin();


    { // Scoping hack in order to have 'scores' defined in each task.


    void* task1 = __hetero_task_begin(
        /* Input Buffers: 4*/ 3, encoded_hv_ptr, encoded_hv_size, clusters_ptr, clusters_size, scores_ptr, scores_size, 
        /* Output Buffers: 1*/ 1,  scores_ptr, scores_size, "clustering_scoring_task"
    );


    __hypervector__<D, hvtype> encoded_hv = *encoded_hv_ptr;
    __hypermatrix__<K, D, hvtype> clusters = *clusters_ptr;

    __hypervector__<K, SCORES_TYPE> scores = *scores_ptr; // Precision of these scores might need to be increased.

    #ifdef HAMMING_DIST
    *scores_ptr =  __hetero_hdc_hamming_distance<K, D, hvtype>(encoded_hv, clusters);
    #else
    *scores_ptr = __hetero_hdc_cossim<K, D, hvtype>(encoded_hv, clusters);
    #endif
    *scores_ptr = __hetero_hdc_absolute_value<K, hvtype>(*scores_ptr);

   __hetero_task_end(task1);
    }
    
    {
   void* task2 = __hetero_task_begin(
        /* Input Buffers: 1*/ 4, encoded_hv_ptr, encoded_hv_size, scores_ptr, scores_size, labels, labels_size,
        /* paramters: 1*/      encoded_hv_idx,
        /* Output Buffers: 1*/ 2, encoded_hv_ptr, encoded_hv_size, labels, labels_size, "find_score_and_label_task"
    );

    
    __hypervector__<K, hvtype> scores = *scores_ptr;
    int max_idx = 0;

    hvtype* elem_ptr = (hvtype*) scores_ptr;

    // IF using hamming distance:
    
    #ifdef HAMMING_DIST
    SCORES_TYPE max_score = (SCORES_TYPE) D - elem_ptr[0]; // I think this is probably causing issues.
    #else
    SCORES_TYPE max_score = (SCORES_TYPE) elem_ptr[0];
    #endif
    
    for (int k = 0; k < K; k++) {
        #ifdef HAMMING_DIST
        SCORES_TYPE score = (SCORES_TYPE) D - elem_ptr[k];
        #else
        SCORES_TYPE score = (SCORES_TYPE) elem_ptr[k];
        #endif
        if (score > max_score) {
            max_score = score;
            max_idx = k;
        }
        
    } 
    // Write labels
    //labels[encoded_hv_idx] = max_idx;
    *labels = max_idx;

    __hetero_task_end(task2);
    }
    __hetero_section_end(section);
    return;
}


// Dimensionality, Clusters, data point vectors, features per.
template <int D, int K>
void root_node( /* Input buffers: 4*/ 
                __hypermatrix__<K, D, hvtype>* clusters_ptr, size_t clusters_size, 
                __hypermatrix__<K, D, hvtype>* temp_clusters_ptr, size_t temp_clusters_size, // ALSO AN OUTPUT
                __hypervector__<D, hvtype>* encoded_hv_ptr, size_t encoded_hv_size, 
                /* Local Vars: 2*/
                __hypervector__<K, SCORES_TYPE>* scores_ptr, size_t scores_size,
                __hypervector__<D, hvtype>* update_hv_ptr, size_t update_hv_size,  
                /* Parameters: 21*/
                int labels_index, 
                /* Output Buffers: 2*/
                int* labels, size_t labels_size){

    void* root_section = __hetero_section_begin();


    // Re-encode each iteration.
    void* inference_task = __hetero_task_begin(
            /* Input Buffers:  */ 7, 
            clusters_ptr,  clusters_size, 
            labels,  labels_size,
            temp_clusters_ptr,  temp_clusters_size, // ALSO AN OUTPUT
            encoded_hv_ptr, encoded_hv_size, 
            scores_ptr, scores_size,
            update_hv_ptr,  update_hv_size,  
            labels_index,  // <- not used.

            /* Output Buffers: 1 */ 2, 
            encoded_hv_ptr, encoded_hv_size,
            labels,  labels_size,
            "inference_task"  
            );

   clustering_node<D, K>(
            encoded_hv_ptr, encoded_hv_size, 
            clusters_ptr,  clusters_size, 
            temp_clusters_ptr,  temp_clusters_size, 
            scores_ptr, scores_size,
            update_hv_ptr,  update_hv_size,  
            labels_index, 
            labels,  labels_size );

    __hetero_task_end(inference_task);


    void* update_task = __hetero_task_begin(
        /* Input Buffers: 4 */  4, 
                                encoded_hv_ptr, encoded_hv_size, 
                                temp_clusters_ptr, temp_clusters_size, 
                                update_hv_ptr, update_hv_size,
                                labels, labels_size,
        /* Output Buffers: 1 */ 2,  
        labels, labels_size,
        temp_clusters_ptr, temp_clusters_size,
        "update_task"  
    );

    {

        *update_hv_ptr =  __hetero_hdc_get_matrix_row<K, D, hvtype>(*temp_clusters_ptr, K, D, *labels);
        *update_hv_ptr = __hetero_hdc_sum<D, hvtype>(*update_hv_ptr, *encoded_hv_ptr); // May need an instrinsic for this.
        __hetero_hdc_set_matrix_row<K, D, hvtype>(*temp_clusters_ptr, *update_hv_ptr, *labels); // How do we normalize?

    }

    __hetero_task_end(update_task);

    __hetero_section_end(root_section);
    return;
}
