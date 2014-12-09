#script to generate all possible answers for a 3x3 grid

####################
# HELPER FUNCTIONS #
####################
def populate(grid, seed):
    for i in range(0, 3):
        for j in range(0, 3):
            bit = j + i * 3
            grid[i][j]= (seed >> bit) % 2

def computeResult(grid):
    temp = 0
    temp += grid[0][0]
    temp += grid[0][1]
    temp += grid[0][2]
    temp += grid[1][0]
    temp += grid[1][2]
    temp += grid[2][0]
    temp += grid[2][1]
    temp += grid[2][2]

    if (((grid[1][1] == 1) and (temp == 2 or temp == 3)) or
         (grid[1][1] == 0) and (temp == 3)):
        return 1

    return 0

def gridAsInt(grid):
    temp = 0
    temp += grid[0][0] << 0
    temp += grid[0][1] << 1
    temp += grid[0][2] << 2
    temp += grid[1][0] << 3
    temp += grid[1][1] << 4
    temp += grid[1][2] << 5
    temp += grid[2][0] << 6
    temp += grid[2][1] << 7
    temp += grid[2][2] << 8
    return temp
 
########
# MAIN #
########
grid = [[0,0,0],[0,0,0],[0,0,0]]
f = open('result', 'w')
for i in range(0, 512):
    populate(grid, i)
    result = computeResult(grid)
    print(gridAsInt(grid))
    f.write(str(result) + ",")
