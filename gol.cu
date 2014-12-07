//// gol.cu 
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
// Game of Life rules

// we define WORD_CEIL because normal ceil() would require using FP
// functional units, which are scarce on the GPU.
#define WORD_CEIL(num, base) (((num - 1) >> 5) + 1)
#define SET_BIT(num, bit, value) (num = ((num & ~(0x1 << bit)) | (value << bit)))
#define GET_BIT(num, bit) ((num >> bit) & 0x1)
typedef unsigned int GolCell;

// global memory only
// # bytes
#define SHARED_MEMORY_SIZE (49152 / sizeof(unsigned int))


// global memory only
inline __device__ void UpdateNeighbourhood(int &neighbourhood, int &neighbourValue) {
	neighbourhood += neighbourValue;
} 

inline __device__ unsigned int GetWord(unsigned int *grid, int x, int y, int gridWidth) {
    int index = (x >> 5) + WORD_CEIL(gridWidth, 32) * y;
    //int bit = x % 32;
    int word  = grid[index];
	return word;
}

inline __device__ unsigned int GetCell(unsigned int *grid, int x, int y, int gridWidth) {
	// Div by 32
    int index = (x >> 5) + WORD_CEIL(gridWidth, 32) * y;
    int bit   = x & (32 - 1);//x % 32;
    int word  = grid[index];
	return GET_BIT(word, bit);
}

inline __device__ void SetCell(unsigned int *grid, int x, int y, int gridWidth, int value) {
    int index = (x >> 5) + WORD_CEIL(gridWidth, 32) * y;
    int bit = x & (32 - 1);

    grid[index] = SET_BIT(grid[index], bit, value);
}

inline __device__ bool IsAlive(int cell) {
	return (1 == cell);
}

// A cell is alive the next generation if it is currently alive and has
// either 2 or 3 neighbours OR if it is dead and has 3 neighbours.
inline __device__ void UpdateState(unsigned int &thisCell, int &neighbourhood) {
	if(IsAlive(thisCell)) {
		thisCell = (neighbourhood == 2 || neighbourhood == 3);
	} else {
		thisCell = (neighbourhood == 3);
	} 
}

__global__
void RunGolSharedMem(unsigned int *g_input, unsigned int *g_output, int gridWidth, int gridHeight, int iterations, bool wrapAround) {
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
    int x = tid_x + blockIdx.x * blockDim.x;
    int y = tid_y + blockIdx.y * blockDim.y;
	int gridSizeX = blockDim.x * gridDim.x;
	int gridSizeY = blockDim.y * gridDim.y;
	if(2 * gridHeight * gridWidth * 8 > SHARED_MEMORY_SIZE * 8) {
		return;
	}

	__shared__ unsigned int buf[SHARED_MEMORY_SIZE];
	unsigned int *input = buf;
	unsigned int *output = &(buf[(WORD_CEIL(gridWidth, 32) * gridHeight)]);
	unsigned int *dummy = &(output[(WORD_CEIL(gridWidth, 32) * gridHeight)]);
	
	int width = WORD_CEIL(gridWidth, 32);
	//printf("%d x %d <- {%d, %d}\n", gridWidth, gridHeight, x, y);
	// Bring the grid into shared memory - only one block can run this kernel per grid
	for(int glbl_y = y; glbl_y < gridHeight; glbl_y = glbl_y + gridSizeY) {
		for(int glbl_x = x/* * 32*/; glbl_x < width/*gridWidth*/; glbl_x = glbl_x + gridSizeX) {
			int idx = glbl_x + glbl_y * width;
			input[idx] = g_input[idx];//GetWord(g_input, glbl_x, glbl_x, gridWidth);//g_input[idx];
			
	//		output[idx] = 1;//"0";
		}
	}
	
	__syncthreads();
	
	for(int iter = 0; iter < iterations; iter = iter + 1) {
		
//			int x_max = (glbl_xx + 32 > gridWidth) ? gridWidth : glbl_xx + 32;
		for(int glbl_y = y; glbl_y < gridHeight; glbl_y = glbl_y + gridSizeY) {
			for(int glbl_x = x ; glbl_x < gridWidth; glbl_x = glbl_x + gridSizeX) {
		//		for(int glbl_x = glbl_xx; glbl_x < x_max; glbl_x++) {
					//Assume row-major here
					//int mapCell = (gridWidth * glbl_y) + glbl_x;
					unsigned int thisCell = GetCell(input, glbl_x, glbl_y, gridWidth);//input[mapCell];
					int neighbourhood = 0;

					// Here we assume that (0,0) is the top left of the grid (although there is)
					// nothing stopping it from being the bottom left.

					int x_left  = (glbl_x == 0)              ? gridWidth - 1  : glbl_x - 1;
					int y_above = (glbl_y == 0)              ? gridHeight - 1 : glbl_y - 1;

					int neighbourValue;
					// TOP LEFT
					neighbourValue = GetCell(input, x_left, y_above, gridWidth);
					UpdateNeighbourhood(neighbourhood, neighbourValue);			 
					// TOP
					neighbourValue = GetCell(input, glbl_x, y_above, gridWidth);
					UpdateNeighbourhood(neighbourhood, neighbourValue); 
					
					int x_right = (glbl_x == gridWidth - 1)  ? 0              : glbl_x + 1;
					// TOP RIGHT
					neighbourValue = GetCell(input, x_right, y_above, gridWidth);
					UpdateNeighbourhood(neighbourhood, neighbourValue); 
					// RIGHT
					neighbourValue = GetCell(input, x_right, glbl_y, gridWidth);
					UpdateNeighbourhood(neighbourhood, neighbourValue); 
					
					int y_below = (glbl_y == gridHeight - 1) ? 0              : glbl_y + 1;
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
					neighbourValue = GetCell(input, x_left, glbl_y, gridWidth);
					UpdateNeighbourhood(neighbourhood, neighbourValue); 

					UpdateState(thisCell, neighbourhood);

					thisCell = __ballot(thisCell);
					int index = (glbl_x >> 5) + WORD_CEIL(gridWidth, 32) * glbl_y;
					if(threadIdx.x  == 0) {
						output[index] = thisCell;
//						printf("%d, %d - %x\n", glbl_x, glbl_y, output[index]);
					}
					//SetCell(output, glbl_x, glbl_y, gridWidth, thisCell);
				}
//			}
		}

		unsigned int *bufferSwap = input;
		input = output;
		output = bufferSwap;
		__syncthreads();
	}
	
	// swap back so output actually points to output data
	unsigned int *bufferSwap = input;
	input = output;
	output = bufferSwap;
	__syncthreads();
	
	// Dump the results back into global memory
	for(int glbl_y = y; glbl_y < gridHeight; glbl_y = glbl_y + gridSizeY) {
		for(int glbl_x = x; glbl_x < width; glbl_x = glbl_x + gridSizeX) {
			int idx = glbl_x + glbl_y * width;
			g_output[idx] = output[idx];//GetWord(output, glbl_x, glbl_y, gridWidth);//output[idx];
//			printf("{%d,%d} IN_G=%x, IN_S=%x\n", glbl_x, glbl_y, g_output[idx], output[idx]);
		}
	}
}

__global__ 
void RunGoL(unsigned int *input, unsigned int *output, int gridWidth, int gridHeight, bool wrapAround) {
	int tid_x = threadIdx.x;
	int tid_y = threadIdx.y;
    int x = tid_x + blockIdx.x * blockDim.x;
    int y = tid_y + blockIdx.y * blockDim.y;
    unsigned int thisCell = 0;

   	// The variable we use to track the status of the cells surrounding this one
	// A basic implementation will be one where for each neighbour that is alive
	// the neighbourhood value increases by one   int thisCell = 0;
	int neighbourhood = 0;

    // TODO: style --> replace 32 with constants or something
    for(int glbl_x = x * 32; glbl_x < x + 32; glbl_x++) {
        neighbourhood = 0;

		// Here we assume that (0,0) is the top left of the grid (although there is)
		// nothing stopping it from being the bottom left.

		int x_left  = (glbl_x == 0) ?             gridWidth - 1  : glbl_x - 1;
		int x_right = (glbl_x == gridWidth - 1) ? 0              : glbl_x + 1;
		int y_above = (y == 0) ?                  gridHeight - 1 : y - 1;
		int y_below = (y == gridHeight - 1) ?     0              : y + 1;
				
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
/*
void InitializeBoard(unsigned int *input, int gridWidth, int gridHeight, char *startingFile) {
	FILE *file = fopen(startingFile, "r");    
	assert(file);
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
*/

void InitializeBoard(unsigned int *input, int gridWidth, int gridHeight, char *startingFile) {
	FILE *file = fopen(startingFile, "r");    
	assert(file);
//	std::cout <<"w,h:"<<gridWidth<<" "<<gridHeight<<std::endl;
	for(int i = 0; i < gridHeight; i = i + 1) {
		for(int j = 0; j < gridWidth; j = j + 1) {
			char cell = fgetc(file);
			
            int index = (j / 32) + WORD_CEIL(gridWidth, 32) * i;
			// Sorry about this - I would like a nicer way to deal with newline oddities across windows/Linux plats
			// but we can hack it for now
			int bit = j % 32;
			while(cell != '1' && cell != 'X' && cell != '0'&& cell != ' ') {
				cell = fgetc(file);
			}
			if((cell == '1' || cell == 'X')) {
				SET_BIT(input[index], bit, '1');
//				input[index] = SET_BIT(input[index], bit, '1');
			} else if((cell == '0' || cell == ' ')) {
				SET_BIT(input[index], bit, '0');
//				input[index] = SET_BIT(input[index], bit, '0');
			}
			
		}
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
	
	unsigned int *input = (unsigned int *)malloc(WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	unsigned int *output = (unsigned int *)malloc(WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	memset (input, 0,  WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	memset (output, 0,  WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	InitializeBoard(input, gridWidth, gridHeight, startingFile);
	
	unsigned int *d_input;
	unsigned int *d_output;
    unsigned int *temp;
	
	cudaMalloc(&d_input, WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	cudaMalloc(&d_output, WORD_CEIL(gridWidth, 32) * gridHeight * 4);
	cudaMemcpy(d_input, input, WORD_CEIL(gridWidth, 32) * gridHeight * 4, cudaMemcpyHostToDevice);

	int width = WORD_CEIL(gridWidth, 32);
//	printf("GOL%d x %d\n", gridWidth, gridHeight);
	// Bring the grid into shared memory - only one block can run this kernel per grid
	for(int glbl_y = 0; glbl_y < gridHeight; glbl_y = glbl_y + 1) {
		for(int glbl_x = 0/* * 32*/; glbl_x < width/*gridWidth*/; glbl_x = glbl_x + 1) {
			int idx = glbl_x + glbl_y * width;
			
//			printf("{%d,%d} IN_G=%x, IN_S=%x\n", glbl_x, glbl_y, input[idx]);

		}
	}
	
    // run GoL for as many iterations as was passed in. note we must repeatedly call
    // the kernel from the host since this is the only way to ensure synchronization
    // across all blocks
	if(gridHeight * gridWidth * 2 * 8 < SHARED_MEMORY_SIZE * 8) {
		int THREADS_X = 32;
		int THREADS_Y = 32;
		int THREADS_Z = 1;
	
		int BLOCKS_MAX = 256;
		int BLOCKS_X = min(BLOCKS_MAX, gridWidth / THREADS_X) + 1;
		int BLOCKS_Y = min(BLOCKS_MAX, gridHeight / THREADS_Y) + 1;
		int BLOCKS_Z = 1;
		dim3 threads(THREADS_X, THREADS_Y, THREADS_Z);
		dim3 blocks(1, 1, 1);
		RunGolSharedMem<<<blocks, threads>>>(d_input, d_output, gridWidth, gridHeight, iterations, true);
	} else {
		int THREADS_X = min(1024, WORD_CEIL(gridWidth, 32));
		int THREADS_Y = min(1024 / THREADS_X, gridHeight);
		int THREADS_Z = 1;
	
		int BLOCKS_X = WORD_CEIL(WORD_CEIL(gridWidth, 32), THREADS_X);
		int BLOCKS_Y = WORD_CEIL(gridHeight, THREADS_Y);
		int BLOCKS_Z = 1;
		dim3 threads(THREADS_X, THREADS_Y, THREADS_Z);
		dim3 blocks(BLOCKS_X, BLOCKS_Y, BLOCKS_Z);
		for (int i = 0; i < iterations; i++) {	
			RunGoL<<<blocks, threads>>>(d_input, d_output, gridWidth, gridHeight, true);
			cudaDeviceSynchronize(); // why can't we move this out of the loop? isn't sync across blocks inherent across kerenl calls?
	
			// swap
			temp = d_input;
			d_input = d_output;
			d_output = temp;
		}
		if((iterations & 0x1)) {
			GolCell *temp = d_input;
			d_input = d_output;
			d_output = temp;
		}
	}

	cudaMemcpy(output, d_output, WORD_CEIL(gridWidth, 32) * gridHeight * 4, cudaMemcpyDeviceToHost);


    // TODO: style	
	for(int j = 0; j < gridHeight; j = j + 1) {
	    for(int i = 0; i < gridWidth; i = i + 1) {
            int index = (i / 32) + WORD_CEIL(gridWidth, 32) * j;
            int bit = i % 32;
			std::cout << (GET_BIT(output[index], bit) ? '#' : ' ');
		}
		std::cout << "|" << std::endl;
	}

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
}

