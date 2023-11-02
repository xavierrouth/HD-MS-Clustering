#define HPVM 1

#ifdef HPVM
#include <heterocc.h>
#include <hpvm_hdc.h>
#include "DFG.hpp"
#endif
#include "host.h"
#include <vector>
#include <cassert>
#include <cmath>


#define HAMMING_DIST
#define OFFLOAD_RP_GEN


#ifdef HAMMING_DIST
#define SCORES_TYPE hvtype
#else
#define SCORES_TYPE float
#endif


#define DUMP(vec, suffix) {\
  FILE *f = fopen("dump/" #vec suffix, "w");\
  if (f) fwrite(vec.data(), sizeof(vec[0]), vec.size(), f);\
  if (f) fclose(f);\
}

template <int N, typename elemTy>
void inline print_hv(__hypervector__<N, elemTy> hv) {
    // TEMPORARILY DISABLED AS THIS CALL BREAKS Code
    return;
    std::cout << "[";
    for (int i = 0; i < N-1; i++) {
        std::cout << hv[0][i] << ", ";
    }
    std::cout << hv[0][N-1] << "]\n";
    return;
}

void datasetBinaryRead(std::vector<int> &data, std::string path){
	std::ifstream file_(path, std::ios::in | std::ios::binary);
	assert(file_.is_open() && "Couldn't open file!");
	int32_t size;
	file_.read((char*)&size, sizeof(size));
	int32_t temp;
	for(int i = 0; i < size; i++){
		file_.read((char*)&temp, sizeof(temp));
		data.push_back(temp);
	}
	file_.close();
}
template <typename T>
T read_encoded_hv(T* input_vector, size_t loop_index_var) {
	//std::cout << ((float*)datapoint_vector)[loop_index_var] << "\n";
	return input_vector[loop_index_var];
}


int main(int argc, char** argv)
{
	__hpvm__init();

	auto t_start = std::chrono::high_resolution_clock::now();
	std::cout << "Main Starting" << std::endl;

	srand(time(NULL));

    assert(argc == 2 && "Expected parameter");
	int EPOCH = std::atoi(argv[1]);
   
	std::vector<int> X_data;
	datasetBinaryRead(X_data, X_data_path);

	std::cout << "Read Data Starting" << std::endl;
	//srand (time(NULL));

	std::cout << "size: " << X_data.size() / Dhv<< std::endl;
	assert(N_FEAT == X_data.size() / Dhv);

	int labels[N_FEAT];
	size_t labels_size = N_FEAT * sizeof(int);

	std::vector<hvtype> tempVec(X_data.begin(), X_data.end());

	
	hvtype* input_vectors = tempVec.data();

	auto t_elapsed = std::chrono::high_resolution_clock::now() - t_start;
	long mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();
	long mSec1 = mSec;
	std::cout << "Reading data took " << mSec << " mSec" << std::endl;

	t_start = std::chrono::high_resolution_clock::now();

	// Host allocated memory 
	__hypervector__<Dhv, hvtype> encoded_hv = __hetero_hdc_hypervector<Dhv, hvtype>();
	hvtype* encoded_hv_buffer = new hvtype[Dhv];
	*((__hypervector__<Dhv, hvtype>*) encoded_hv_buffer) = encoded_hv;
	size_t encoded_hv_size = Dhv * sizeof(hvtype);

	hvtype* update_hv_ptr = new hvtype[Dhv];
	size_t update_hv_size = Dhv * sizeof(hvtype);
	
	// Used to store a temporary cluster for initializion
	__hypervector__<Dhv, hvtype> cluster = __hetero_hdc_hypervector<Dhv, hvtype>();
	hvtype* cluster_buffer = new hvtype[Dhv];
	size_t cluster_size = Dhv * sizeof(hvtype);

	// Read from during clustering, updated from clusters_temp.
	__hypermatrix__<N_CENTER, Dhv, hvtype> clusters = __hetero_hdc_hypermatrix<N_CENTER, Dhv, hvtype>();
	hvtype* clusters_buffer = new hvtype[N_CENTER * Dhv];
	size_t clusters_size = N_CENTER * Dhv * sizeof(hvtype);

	// Gets written into during clustering, then is used to update 'clusters' at the end.
	__hypermatrix__<N_CENTER, Dhv, hvtype> clusters_temp = __hetero_hdc_hypermatrix<N_CENTER, Dhv, hvtype>();
	hvtype* clusters_temp_buffer = new hvtype[N_CENTER * Dhv];

	// Temporarily store scores, allows us to split score calcuation into a separte task.


	__hypervector__<Dhv, SCORES_TYPE> scores = __hetero_hdc_hypervector<Dhv, SCORES_TYPE>();
	SCORES_TYPE* scores_buffer = new SCORES_TYPE[N_CENTER];
	size_t scores_size = N_CENTER * sizeof(SCORES_TYPE);



	std::cout << "Dimension over 32: " << Dhv/32 << std::endl;
	//We need a seed ID. To generate in a random yet determenistic (for later debug purposes) fashion, we use bits of log2 as some random stuff.

	std::cout << "Seed hv:\n";
	std::cout << "After seed generation\n";

	// ========== Initialize cluster hvs =============== .
	std::cout << "Init cluster hvs:" << std::endl;
	for (int k = 0; k < N_CENTER; k++) {
		__hypervector__<Dhv, hvtype> encoded_cluster_hv = __hetero_hdc_create_hypervector<Dhv, hvtype>(1, (void*) read_encoded_hv<hvtype>, input_vectors + k * ENCODED_HV_SIZE_PAD);

		std::cout <<" Cluster "<< k << "\n";

        //print_hv<Dhv, hvtype>(cluster);
		__hetero_hdc_set_matrix_row<N_CENTER, Dhv, hvtype>(clusters, encoded_cluster_hv, k);
	}


	std::cout << "\nDone init cluster hvs:" << std::endl;

	#if DEBUG
	for (int i = 0; i < N_CENTER; i++) {
		__hypervector__<Dhv, hvtype> cluster_temp = __hetero_hdc_get_matrix_row<N_CENTER, Dhv, hvtype>(clusters, N_CENTER, Dhv, i);
		std::cout << i << " ";
		print_hv<Dhv, hvtype>(cluster_temp);
	}
	#endif

    int label_index = 1;

	// =================== Clustering ===============================
	for (int i = 0; i < EPOCH; i++) {
		// Can we normalize the hypervectors here or do we have to do that in the DFG.
		std::cout << "Epoch: #" << i << std::endl;
		for (int j = 0; j < N_SAMPLE; j++) {

			__hypervector__<Dhv, hvtype> encoded_cluster_hv = __hetero_hdc_create_hypervector<Dhv, hvtype>(1, (void*) read_encoded_hv<hvtype>, input_vectors + j * ENCODED_HV_SIZE_PAD);

			// Root node is: Encoding -> Clustering for a single HV.
			void *DFG = __hetero_launch(
				(void*) root_node<Dhv, N_CENTER>,
				/* Input Buffers: 4*/ 7,
				&clusters, clusters_size, //false,
				&clusters_temp, clusters_size, //false,
				encoded_hv_buffer, encoded_hv_size,// false,
				scores_buffer, scores_size,
                update_hv_ptr, update_hv_size,
				j, 
				/* Output Buffers: 1*/ 

                // Directly just push the pointer offset for the location to update
				(labels+j), sizeof(int),

				2,
                // Directly just push the pointer offset for the location to update
				(labels+j), sizeof(int),

				&clusters_temp, clusters_size
			);
			__hetero_wait(DFG); 

			//std::cout << "after root launch" << std::endl;
		
		}
		// then update clusters and copy clusters_tmp to clusters, 

	}


	t_elapsed = std::chrono::high_resolution_clock::now() - t_start;
	
	mSec = std::chrono::duration_cast<std::chrono::milliseconds>(t_elapsed).count();
	
	std::cout << "\nReading data took " << mSec1 << " mSec" << std::endl;    
	std::cout << "Execution (" << EPOCH << " epochs)  took " << mSec << " mSec" << std::endl;
	
	std::ofstream myfile("out.txt");
	for(int i = 0; i < N_SAMPLE; i++){
		myfile << " " << labels[i] << std::endl;
	}
	__hpvm__cleanup();
	return 0;
}




