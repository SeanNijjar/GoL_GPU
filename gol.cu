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

// global memory only

typedef bool GolCell;

// TODO: this function is unused?
/*
inline __device__ GolCell GetNeighbourCell (GolCell *input, int mapCellIdx, int mapWidth, int x_off, int y_off) {
	return input[mapCellIdx + (mapWidth * y_off) + x_off];
}
*/

inline __device__ void UpdateNeighbourhood(int &neighbourhood, int &neighbourValue) {
	neighbourhood += neighbourValue;
} 

inline __device__ int GetCell(int *grid, int x, int y, int gridWidth) {
    int index = (x / 32) + WORD_CEIL(gridWidth, 32) * y;
    int bit = x % 32;
    int word = grid[index];
	return GET_BIT(word, bit);
}

inline __device__ void SetCell(int *grid, int x, int y, int gridWidth, int value) {
    int index = (x / 32) + WORD_CEIL(gridWidth, 32) * y;
    int bit = x % 32;
    grid[index] = SET_BIT(grid[index], bit, value);
}

inline __device__ bool IsAlive(int cell) {
	return (1 == cell);
}

// A cell is alive the next generation if it is currently alive and has
// either 2 or 3 neighbours OR if it is dead and has 3 neighbours.
inline __device__ void UpdateState(int &thisCell, int &neighbourhood) {
	if(IsAlive(thisCell)) {
		thisCell = (neighbourhood == 2 || neighbourhood == 3);
	} else {
		thisCell = (neighbourhood == 3);
	} 
}

__global__ 
void RunGoL(int *input, int *output, int gridWidth, int gridHeight, bool wrapAround) {
    //extern __shared__ bool s_data[];

	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
    int x = tid_x + blockIdx.x * blockDim.x;
    int y = tid_y + blockIdx.y * blockDim.y;
    int thisCell = 0;

   	// The variable we use to track the status of the cells surrounding this one
	// A basic implementation will be one where for each neighbour that is alive
	// the neighbourhood value increases by one   int thisCell = 0;
	int neighbourhood = 0;

    // TODO: style --> replace 32 with constants or something
    for(int glbl_x = x; glbl_x < x + 32; glbl_x++) {
        neighbourhood = 0;

		// Here we assume that (0,0) is the top left of the grid (although there is)
		// nothing stopping it from being the bottom left.

		int x_left = (glbl_x == 0) ? gridWidth - 1 : glbl_x - 1;
		int x_right = (glbl_x == gridWidth - 1) ? 0 : glbl_x + 1;
		int y_above = (y == 0) ? gridHeight - 1 : y - 1;
		int y_below = (y == gridHeight - 1) ? 0 : y + 1;
				
		int neighbourValue;
		// TOP LEFT
		neighbourValue = GetCell(input, x_left, y_above, gridWidth);
		UpdateNeighbourhood(neighbourhood, neighbourValue);
			 
		// TOP
		neighbourValue = GetCell(input, glbl_x, y_above, gridWidth);
		UpdateNeighbourhood(neighbourhood, neighbourValue); 

		// TOP RIGHT
		neighbourValue = GetCell(input, x_right, y_above, gridWidth);
		UpdateNeighbourhood(neighbourhood, neighbourValue); 

		// RIGHT
		neighbourValue = GetCell(input, x_right, y, gridWidth);
		UpdateNeighbourhood(neighbourhood, neighbourValue); 

		// BOTTOM RIGHT
		neighbourValue = GetCell(input, x_right, y_below, gridWidth);
		UpdateNeighbourhood(neighbourhood, neighbourValue); 

		// BOTTOM
		neighbourValue = GetCell(input, glbl_x, y_below, gridWidth);
		UpdateNeighbourhood(neighbourhood, neighbourValue); 
				
		// BOTTOM LEFT
		neighbourValue = GetCell(input, x_left, y_below, gridWidth);
		UpdateNeighbourhood(neighbourhood, neighbourValue); 

		// LEFT
		neighbourValue = GetCell(input, x_left, y, gridWidth);
		UpdateNeighbourhood(neighbourhood, neighbourValue); 

        // update the state
        thisCell = GetCell(input, glbl_x, y, gridWidth);
        UpdateState(thisCell, neighbourhood);
        SetCell(output, glbl_x, y, gridWidth, thisCell);
	}
} 

void InitializeBoard(int *input, int gridWidth, int gridHeight, char *startingFile) {
	FILE *file = fopen(startingFile, "r");    
	assert(file);
//	std::cout <<"w,h:"<<gridWidth<<" "<<gridHeight<<std::endl;
  	for(int i = 0; i < gridHeight; i = i + 1) {
    	for(int j = 0; j < gridWidth; j = j + 1) {
			char cell = fgetc(file);
            int index = (j / 32) + WORD_CEIL(gridWidth, 32) * i;
            int bit = j % 32;
			input[index] = SET_BIT(input[index], bit, (cell == '1'));
			std::cout << GET_BIT(input[index], bit);
		}
        std::cout << std::endl;
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
	int gridSize  = gridWidth * gridHeight;
	char *startingFile = argv[4];
	
	int *input = (int *)malloc(WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	int *output = (int *)malloc(WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	
	InitializeBoard(input, gridWidth, gridHeight, startingFile);
	
	int THREADS_X = min(1024, WORD_CEIL(gridWidth, 32));
	int THREADS_Y = min(1024 / THREADS_X, gridHeight);
	//int THREADS_Y = 1;
    //int THREADS_Y = ;
	int THREADS_Z = 1;
	
	int BLOCKS_X = WORD_CEIL(WORD_CEIL(gridWidth, 32), THREADS_X);
	int BLOCKS_Y = WORD_CEIL(gridHeight, THREADS_Y);
    //int BLOCKS_Y = 1024 / ;
	int BLOCKS_Z = 1;
	
	dim3 threads(THREADS_X, THREADS_Y, THREADS_Z);
	dim3 blocks(BLOCKS_X, BLOCKS_Y, BLOCKS_Z);
	
	int *d_input;
	int *d_output;
    int *temp;
    // TODO: style
	cudaMalloc(&d_input, WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	cudaMalloc(&d_output, WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	cudaMemcpy(d_input, input, WORD_CEIL(gridWidth, 32) * gridHeight * 4, cudaMemcpyHostToDevice);

    // run GoL for as many iterations as was passed in. note we must repeatedly call
    // the kernel from the host since this is the only way to ensure synchronization
    // across all blocks
    for (int i = 0; i < iterations; i++) {	
	    RunGoL<<<blocks, threads>>>(d_input, d_output, gridWidth, gridHeight, true);
        cudaDeviceSynchronize(); // why can't we move this out of the loop? isn't sync across blocks inherent across kerenl calls?

        // swap
        temp = d_input;
        d_input = d_output;
        d_output = temp;
    }	

	cudaMemcpy(output, d_input, WORD_CEIL(gridWidth, 32) * gridHeight * 4, cudaMemcpyDeviceToHost);

    std::cout << std::endl;
    // TODO: style	
	for(int j = 0; j < gridHeight; j = j + 1) {
	    for(int i = 0; i < gridWidth; i = i + 1) {
            int index = (i / 32) + WORD_CEIL(gridWidth, 32) * j;
            int bit = i % 32;
			std::cout << GET_BIT(output[index], bit);
		}
		std::cout << std::endl;
	}

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
}

