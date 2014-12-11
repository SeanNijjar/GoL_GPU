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

    __constant__ unsigned int lookUp [] = {0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};   
// there are 32 bits in a GPU word
#define W_SIZE (32)

// gets a group of 3 bits centered around "bit".
#define GET_BITS(value, bit) (((value) >> ((bit) - 1)) & 0x7)

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

// global memory only
inline __device__ void UpdateNeighbourhood(int &neighbourhood, int &neighbourValue) {
    neighbourhood += neighbourValue;
} 

inline __device__ unsigned int GetWord(unsigned int *grid, int x, int y, int gridWidth) {
    int index = (x >> 5) + WORD_CEIL(gridWidth, 32) * y;
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

// TODO: would this be more efficiently implemented as unsigned int or as unsigned char?
// TODO: would this cause bank conflicts?
__device__ unsigned int ComputeResult(int value) {
    // look up table for 3 X 3 grid of values. the top left corner is the MSB
    // and the bottom right corner is the LSB:
    //
    // MSB, X, X
    // X  , X, X
    // X  , X, LSB
    //const unsigned int lookUp [] = {0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};   
 
    return lookUp[value];
}

__global__
void RunGolSharedMem(unsigned int *g_input, unsigned int *g_output, int gridWidth, int gridHeight, int iterations) {
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
    // Bring the grid into shared memory - only one block can run this kernel per grid
    for(int glbl_y = y; glbl_y < gridHeight; glbl_y = glbl_y + gridSizeY) {
        for(int glbl_x = x/* * 32*/; glbl_x < width/*gridWidth*/; glbl_x = glbl_x + gridSizeX) {
            int idx = glbl_x + glbl_y * width;
            input[idx] = g_input[idx];//GetWord(g_input, glbl_x, glbl_x, gridWidth);//g_input[idx];
        }
    }
    
    __syncthreads();
    
    for(int iter = 0; iter < iterations; iter = iter + 1) {
        for(int glbl_y = y; glbl_y < gridHeight; glbl_y = glbl_y + gridSizeY) {
            for(int glbl_x = x ; glbl_x < gridWidth; glbl_x = glbl_x + gridSizeX) {
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

                int index = (glbl_x >> 5) + WORD_CEIL(gridWidth, 32) * glbl_y;

                if(threadIdx.x  == 0) {
                    output[index] = thisCell;
                }
            }
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
        }
    }
}

// TODO: implement smem
// TODO: need to find optimum number of "center words"
__global__ 
void RunGoL(unsigned int *input, unsigned int *output, int gridWidth, int gridHeight) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int wordsPerRow = WORD_CEIL(gridWidth, W_SIZE);

    if (x >= gridWidth || y >= gridWidth) {
        return;
    }

    // get the target word and all surrounding words
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
    unsigned int end = (wordsPerRow - 1 == x) ? (gridWidth % W_SIZE) - 1 : 31;

    // iterate through 30 bits of the 32 bit word and lookup the result
    for (int i = end - 1; i > 0; i--) {
        // represent the 3X3 grid centered around mid[i] as a 9 bit number
        seed = (GET_BITS(nWord, i) << 6) + (GET_BITS(cWord, i) << 3) + GET_BITS(sWord, i);

        // look up the answer and add it to "result", our intermediate result
        result += ComputeResult(seed) << i;
    }


    // compute right end
    seed = (GET_BIT(nWord, 1) << 8) + (GET_BIT(nWord, 0) << 7) + (GET_BIT(neWord, end) << 6)
         + (GET_BIT(cWord, 1) << 5) + (GET_BIT(cWord, 0) << 4) + (GET_BIT(eWord, end) << 3)
         + (GET_BIT(sWord, 1) << 2) + (GET_BIT(sWord, 0) << 1) + (GET_BIT(seWord, end) << 0);
    result += ComputeResult(seed) << 0;

    // compute left end
    seed = (GET_BIT(nwWord, 0) << 8) + (GET_BIT(nWord, end) << 7) + (GET_BIT(nWord, end - 1) << 6) 
         + (GET_BIT( wWord, 0) << 5) + (GET_BIT(cWord, end) << 4) + (GET_BIT(cWord, end - 1) << 3) 
         + (GET_BIT(swWord, 0) << 2) + (GET_BIT(sWord, end) << 1) + (GET_BIT(sWord, end - 1) << 0) ;
    result += ComputeResult(seed) << end;

    // write out the final answer
    output[x + y * wordsPerRow] = result;
}

void InitializeBoard(unsigned int *input, int gridWidth, int gridHeight, char *startingFile, bool bGenGridFromScratch) {
    FILE *file = NULL;
	if(!bGenGridFromScratch) {
		fopen(startingFile, "r");    
    	assert(file);
	}

    for(int i = 0; i < gridHeight; i = i + 1) {
        for(int j = 0; j < gridWidth; j = j + 1) {
            char cell = '\n';
			if(!bGenGridFromScratch) {
				fgetc(file);
			} else {
				cell = (rand() % 3  == 0) ? '1' : '0'; 
			}
            
            int index = (j / 32) + WORD_CEIL(gridWidth, 32) * i;
            int bit = j % 32;

            // Sorry about this - I would like a nicer way to deal with newline
            // oddities across windows/Linux plats but we can hack it for now
            while(cell != '1' && cell != 'X' && cell != '0'&& cell != ' ' && cell != '_' && !bGenGridFromScratch) {
                cell = fgetc(file);
            }

            if((cell == '1' || cell == 'X')) {
                SET_BIT(input[index], bit, 1);
            } else if((cell == '0' || cell == ' ' || cell == '_')) {
                SET_BIT(input[index], bit, 0);
            }
        }
    }
	if(!bGenGridFromScratch) {
    	fclose(file);
	}
}

int main (int argc, char *argv[]) {
    // sanity checking of parameters
    if(argc != 5 && argc != 4) {
        printf("Usage: gol <gridWidth> <gridHeight> <iterations> <starting file>\n");
        return 0;
    }
   
    // parse in parameters to get dimensions of the board
    int gridWidth = atoi(argv[1]);
    int gridHeight = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    char *startingFile = argv[4];
    int gridSize = WORD_CEIL(gridWidth, W_SIZE) * gridHeight * 4;
   
    // allocate all memory for the boards 
    unsigned int *input = (unsigned int *)malloc(gridSize);
    unsigned int *output = (unsigned int *)malloc(gridSize);
    unsigned int *d_input;
    unsigned int *d_output;
    memset(input, 0, gridSize);
    memset(output, 0, gridSize);
    cudaMalloc(&d_input, gridSize);
    cudaMalloc(&d_output, gridSize);

    // load in the starting condition of the GoL board   
    InitializeBoard(input, gridWidth, gridHeight, startingFile, argc == 4);
	// Cuda Events
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
 
	cudaEventRecord(start, 0);
    // copy the board to video memory in preparation for GPU
    // to do processing
    cudaMemcpy(d_input, input, gridSize, cudaMemcpyHostToDevice);

    // run GoL for as many iterations as was passed in. note we must repeatedly call
    // the kernel from the host since this is the only way to ensure synchronization
    // across all blocks
    if(gridHeight * gridWidth * 2 * 8 < SHARED_MEMORY_SIZE * 8) {
        int THREADS_X = 32;
        int THREADS_Y = 32;
        int THREADS_Z = 1;
        dim3 threads(THREADS_X, THREADS_Y, THREADS_Z);
        dim3 blocks(1, 1, 1);

        RunGolSharedMem<<<blocks, threads>>>(d_input, d_output, gridWidth, gridHeight, iterations);

    } else {
        int THREADS_X = min(1024, WORD_CEIL(gridWidth, 32));
        int THREADS_Y = min(1024 / THREADS_X, gridHeight);
        int THREADS_Z = 1;
    
        int BLOCKS_X = WORD_CEIL(WORD_CEIL(gridWidth, 32), THREADS_X);
        int BLOCKS_Y = WORD_CEIL(gridHeight, THREADS_Y);
        int BLOCKS_Z = 1;
        dim3 threads(THREADS_X, THREADS_Y, THREADS_Z);
        dim3 blocks(BLOCKS_X, BLOCKS_Y, BLOCKS_Z);

        unsigned int *temp;

        for (int i = 0; i < iterations; i++) {  
            RunGoL<<<blocks, threads>>>(d_input, d_output, gridWidth, gridHeight);

            // why can't we move this out of the loop?
            //isn't sync across blocks inherent across kerenl calls?
            cudaDeviceSynchronize();

            // swap input and output GoL boards
            temp = d_input;
            d_input = d_output;
            d_output = temp;
        }

        temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    // copy back everything
    cudaMemcpy(output, d_output, gridSize, cudaMemcpyDeviceToHost);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float time_ms;
	cudaEventElapsedTime(&time_ms, start, end);
	
    // print out the results
//    for(int i = 0; i < gridHeight; i = i + 1) {
//        for(int j = 0; j < gridWidth; j = j + 1) {
//            int index = (j / 32) + WORD_CEIL(gridWidth, 32) * i;
//            int bit = j % 32;
//
//            std::cout << (GET_BIT(output[index], bit) ?  '#' : ' ');
//        }
//        std::cout << std::endl;
//    }
	std::cout <<"time: "<<time_ms<<std::endl;

    // free all resources
    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
}

