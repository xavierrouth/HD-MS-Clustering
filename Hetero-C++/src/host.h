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
#define Dhv				2048  //hypervectors length
#define N_SAMPLE 		258

#define COL				8 //number of columns of a matrix-vector multiplication window
#define ROW				32 //number of rows of a matrix-vector multiplication window (32, 64, 128, 256, 512)

//int EPOCH = 10;
bool shuffled = false;
std::string X_data_path = "../dataset/subset_spectra_hvs_1468.bin";
std::string y_data_path = "../dataset/massspec-labels.txt";

