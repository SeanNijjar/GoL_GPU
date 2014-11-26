//// gol.cu 
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
// Game of Life rules


// global memory only

typedef bool GolCell;

inline GolCell GetNeighbourCell (GolCell *input, int mapCellIdx, int mapWidth, int x_off, int y_off) {
	return input[mapCellIdx + (mapWidth * y_off) + x_off];
}

inline  void UpdateNeighbourhood(int &neighbourhood, GolCell &neighbourValue) {
	neighbourhood += neighbourValue;
} 

inline  GolCell GetCell(GolCell *grid, int x, int y, int gridWidth) {
	return grid[x + (y * gridWidth)];
}

inline  bool IsAlive(GolCell &cell) {
	return (true == cell);
}

// A cell is alive the next generation if it is currently alive and has
// either 2 or 3 neighbours OR if it is dead and has 3 neighbours.
inline void UpdateState(GolCell &thisCell, int &neighbourhood) {
	if(IsAlive(thisCell)) {
		thisCell = (neighbourhood == 2 || neighbourhood == 3);
	} else {
		thisCell = (neighbourhood == 3);
	} 
}


void RunGoL(GolCell *input, GolCell *output, int gridWidth, int gridHeight, int iterations, bool wrapAround) {
	for(int iter = 0; iter < iterations; iter = iter + 1) {
		for(int glbl_x = 0; glbl_x < gridWidth; glbl_x = glbl_x + 1) {
			for(int glbl_y = 0; glbl_y < gridHeight; glbl_y = glbl_y + 1) {
				//Assume row-major here
				int mapCell = (gridWidth * glbl_y) + glbl_x;
				GolCell thisCell = input[mapCell];
				// The variable we use to track the status of the cells surrounding this one
				// A basic implementation will be one where for each neighbour that is alive
				// the neighbourhood value increases by one
				int neighbourhood = 0;
				
				// As is right now, this is a lot of overhead, but I wrote it
				// like this so we can easily add in optis later. At the end,
				// if the CUDA compiler does not do inlining for us, we can manually
				// do inlining of these functions.

				// Here we assume that (0,0) is the top left of the grid (although there is)
				// nothing stopping it from being the bottom left.
				// **JUSTIN** - let me know if you have preference -> it doesn't bother me either way

				int x_left = (glbl_x == 0) ? gridWidth - 1 : glbl_x - 1;
				int x_right = (glbl_x == gridWidth - 1) ? 0 : glbl_x + 1;
				int y_above = (glbl_y == 0) ? gridHeight - 1 : glbl_y - 1;
				int y_below = (glbl_y == gridHeight - 1) ? 0 : glbl_y + 1;
				
				GolCell neighbourValue;
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
				neighbourValue = GetCell(input, x_right, glbl_y, gridWidth);
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
				neighbourValue = GetCell(input, x_left, glbl_y, gridWidth);
				UpdateNeighbourhood(neighbourhood, neighbourValue); 

				UpdateState(thisCell, neighbourhood);

				output[mapCell] = thisCell;

			}
		}
		GolCell *bufferSwap = input;
		input = output;
		output = bufferSwap;
	}
} 

void InitializeBoard(GolCell *input, int gridWidth, int gridHeight, char *startingFile) {
	FILE *file = fopen(startingFile, "r");    
	assert(file);
//	std::cout <<"w,h:"<<gridWidth<<" "<<gridHeight<<std::endl;
	for(int i = 0; i < gridHeight; i = i + 1) {
		for(int j = 0; j < gridWidth; j = j + 1) {
			char cell = fgetc(file);
			
			// Sorry about this - I would like a nicer way to deal with newline oddities across windows/Linux plats
			// but we can hack it for now
			while(cell != '1' && cell != 'X' && cell != '0'&& cell != ' ') {
				cell = fgetc(file);
			}
			if((cell == '1' || cell == 'X')) {
				input[gridWidth * i + j] = 1;
			} else if((cell == '0' || cell == ' ')) {
				input[gridWidth * i + j] = 0;
			}
			
		}
	}
	fclose(file);
}

int main (int argc, char *argv[]) {
	if(argc != 5) {
		return 0;
	}
	
	int gridWidth = atoi(argv[1]);
	int gridHeight = atoi(argv[2]);
	int iterations = atoi(argv[3]);
	int gridSize  = gridWidth * gridHeight;
	char *startingFile = argv[4];
		
	GolCell *input = (GolCell *)malloc(gridSize * sizeof(GolCell));
	GolCell *output = (GolCell *)malloc(gridSize * sizeof(GolCell));
	
	InitializeBoard(input, gridWidth, gridHeight, startingFile);

	RunGoL(input, output, gridWidth, gridHeight, iterations, true);
	
	if(0 == (iterations & 0x1)) { // even number of iterations -> true output buffer is stored at input
		output = input;
	}
//	std::cout <<"Ending BOARD"<<std::endl;
	for(int j = 0; j < gridHeight; j = j + 1) {
		for(int i = 0; i < gridWidth; i = i + 1) {
			std::cout << (output[j * gridWidth + i] ? '#' : ' ');
		}
		std::cout << std::endl;
	}
}

