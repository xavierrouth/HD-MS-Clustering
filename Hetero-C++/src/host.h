#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <chrono>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>

#define N_CENTER		38	//number of centers. (e.g., isolet: 26,)
#define ENCODED_HV_SIZE_PAD 2048
#define Dhv				2048  //hypervectors length
#define N_SAMPLE 		6238
#define N_FEAT 			258

#define COL				8 //number of columns of a matrix-vector multiplication window
#define ROW				32 //number of rows of a matrix-vector multiplication window (32, 64, 128, 256, 512)

#define PAD_			(N_FEAT & (COL - 1))
#if PAD_ == 0
	#define PAD 		0
#else
	#define PAD 		(COL - PAD_)
#endif

#define N_FEAT_PAD		(N_FEAT + PAD)	//feature per input (e.g., isolet: 624, ucihar 568)

//int EPOCH = 10;
bool shuffled = false;
std::string X_data_path = "../dataset/subset_spectra_hvs_1468.bin";
std::string y_data_path = "../dataset/isolet_train_trainY.bin";

