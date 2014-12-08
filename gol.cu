//// gol.cu 
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
// Game of Life rules

// we define WORD_CEIL because normal ceil() would require using FP
// functional units, which are scarce on the GPU.
#define WORD_CEIL(num, base) (((num) + (base) - 1) / (base))
#define SET_BIT(num, bit, value) (num ^ ((value & 0x1) << bit))
#define GET_BIT(num, bit) ((num & (0x1 << bit)) >> bit)

// there are 32 bits in a GPU word
#define W_SIZE (32)

// gets a group of 3 bits centered around "bit".
#define GET_BITS(value, bit) (((value) >> ((bit) - 1)) & 0x7)

inline __device__ unsigned int getModDim(int x, int dim){
    if (x >= 0) {
        return x % dim;
    }
    else {
        return dim + x; // note that x is negative, so this results in a subtraction
    }
}

inline
__device__ unsigned int getNeighbour(unsigned int * input, int x, int y, int width, int height) {
    int index = getModDim(x, width) + getModDim(y, height) * width;
    return input[index];
}

// TODO: would this be more efficiently implemented as unsigned int or as unsigned char?
// TODO: would this cause bank conflicts?
__device__ unsigned int ComputeResult(int value) {
    // look up table for 3 X 3 grid of values. the top left corner is the MSB
    // and the bottom right corner is the LSB:
    //
    // MSB, X, X
    // X  , X, X
    // X  , X, LSB
    const unsigned int lookUp [] = {0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};   
 
    return lookUp[value];
}

__global__ 
void RunGoL(unsigned int *input, unsigned int *output, int gridWidth, int gridHeight, bool wrapAround) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int wordsPerRow = WORD_CEIL(gridWidth, W_SIZE);

    if (x >= gridWidth || y >= gridWidth) {
        return;
    }

    unsigned int nwWord = getNeighbour(input, x - 1, y - 1, wordsPerRow, gridHeight);
    unsigned int nWord  = getNeighbour(input,     x, y - 1, wordsPerRow, gridHeight);
    unsigned int neWord = getNeighbour(input, x + 1, y - 1, wordsPerRow, gridHeight);
    unsigned int wWord  = getNeighbour(input, x - 1,     y, wordsPerRow, gridHeight);
    unsigned int cWord  = getNeighbour(input,     x,     y, wordsPerRow, gridHeight);
    unsigned int eWord  = getNeighbour(input, x + 1,     y, wordsPerRow, gridHeight);
    unsigned int swWord = getNeighbour(input, x - 1, y + 1, wordsPerRow, gridHeight);
    unsigned int sWord  = getNeighbour(input,     x, y + 1, wordsPerRow, gridHeight);
    unsigned int seWord = getNeighbour(input, x + 1, y + 1, wordsPerRow, gridHeight);

    unsigned int seed = 0;
    unsigned int result = 0;

    // iterate through 30 bits of the 32 bit word and lookup the result
    for (int i = W_SIZE - 2; i > 0; i--) {
        // represent the 3X3 grid centered around mid[i] as a 9 bit number
        seed = (GET_BITS(nWord, i) << 6) + (GET_BITS(cWord, i) << 3) + GET_BITS(sWord, i);

        // look up the answer and add it to "result", our intermediate result
        result += ComputeResult(seed) << i;
    }

    // compute the ends
    // TODO: would this be better placed when we retrieve
    // top/mid/bot to guarentee no bank collisions?
    seed = (GET_BIT(nWord, 1) << 8) + (GET_BIT(nWord, 0) << 7) + (GET_BIT(neWord, 31) << 6)
         + (GET_BIT(cWord, 1) << 8) + (GET_BIT(cWord, 0) << 7) + (GET_BIT(eWord, 31) << 6)
         + (GET_BIT(sWord, 1) << 8) + (GET_BIT(sWord, 0) << 7) + (GET_BIT(seWord, 31) << 6);
    result += ComputeResult(seed);

    seed = (GET_BIT(nWord, 31) << 8) + (GET_BIT(nWord, 30) << 7) + (GET_BIT(nwWord, 0) << 6)
         + (GET_BIT(cWord, 31) << 8) + (GET_BIT(cWord, 30) << 7) + (GET_BIT(wWord, 0) << 6)
         + (GET_BIT(sWord, 31) << 8) + (GET_BIT(sWord, 30) << 7) + (GET_BIT(swWord, 0) << 6);
    result += ComputeResult(seed) << 31;

    // write out the final answer
    output[x + y * wordsPerRow] = result;
} 

void InitializeBoard(unsigned int *input, int gridWidth, int gridHeight, char *startingFile) {
	FILE *file = fopen(startingFile, "r");    
	assert(file);
    int wordsPerRow = WORD_CEIL(gridWidth, W_SIZE);

    // read in the file
  	for(int row = 0; row < gridHeight; row++) {
    	for(int col = 0; col < gridWidth; col++) {
			char cell = fgetc(file);
            int word = row * wordsPerRow + col / W_SIZE;
            int bit = 31 - col % W_SIZE;
			input[word] = SET_BIT(input[word], bit, cell == 'X' || cell == '1');
			//std::cout << GET_BIT(input[word], bit);
		}
        //std::cout << std::endl;
	}
	fclose(file);
}

int main (int argc, char *argv[]) {
	if(argc != 5) {
        printf("Usage: gol <gridWidth> <gridHeight> <iterations> <starting file>\n");
		return 0;
	}
	
	int gridWidth = atoi(argv[1]);
	int gridHeight = atoi(argv[2]);
	int iterations = atoi(argv[3]);
	char *startingFile = argv[4];
    int gridSize = WORD_CEIL(gridWidth, W_SIZE) * gridHeight * 4;
	
	unsigned int *input = (unsigned int *)malloc(gridSize);
	unsigned int *output = (unsigned int *)malloc(gridSize);
    memset(input, 0, gridSize);
    memset(output, 0, gridSize);
	
	InitializeBoard(input, gridWidth, gridHeight, startingFile);
	
	int THREADS_X = min(1024, WORD_CEIL(gridWidth, W_SIZE));
	int THREADS_Y = min(1024 / THREADS_X, gridHeight);
	int THREADS_Z = 1;
	
	int BLOCKS_X = WORD_CEIL(WORD_CEIL(gridWidth, W_SIZE), THREADS_X);
	int BLOCKS_Y = WORD_CEIL(gridHeight, THREADS_Y);
	int BLOCKS_Z = 1;
	
	dim3 threads(THREADS_X, THREADS_Y, THREADS_Z);
	dim3 blocks(BLOCKS_X, BLOCKS_Y, BLOCKS_Z);

	unsigned int *d_input;
	unsigned int *d_output;
    unsigned int *temp;

	cudaMalloc(&d_input, gridSize);
	cudaMalloc(&d_output, gridSize);
	cudaMemcpy(d_input, input, gridSize, cudaMemcpyHostToDevice);

    // run GoL for as many iterations as was passed in. note we must repeatedly call
    // the kernel from the host since this is the only way to ensure synchronization
    // across all blocks
    for (int i = 0; i < iterations; i++) {
	    RunGoL<<<blocks, threads>>>(d_input, d_output, gridWidth, gridHeight, true);
        cudaDeviceSynchronize(); // TODO: is this necessary

        // swap
        temp = d_input;
        d_input = d_output;
        d_output = temp;
    }	

	cudaMemcpy(output, d_input, gridSize, cudaMemcpyDeviceToHost);

    //std::cout << std::endl;
    int wordsPerRow = WORD_CEIL(gridWidth + 2, W_SIZE);
  	for(int row = 0; row < gridHeight; row++) {
    	for(int col = 0; col < gridWidth; col++) {
            int word = row * wordsPerRow + col / W_SIZE;
            int bit = 31 - col % W_SIZE;
			std::cout << (GET_BIT(output[word], bit) ? '#' : ' ');
		}
        std::cout << std::endl;
	}

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
}

