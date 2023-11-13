#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

using namespace std;


class Sudoku
{
private:
    const int UNASSIGNED = 0;
    int N;
    int **grid;

    bool FindUnassignedLocation(int &row, int &col)
    {
        for (row = 0; row < N; row++)
            for (col = 0; col < N; col++)
                if (grid[row][col] == UNASSIGNED)
                    return true;
        return false;
    }

    bool UsedInRow(int row, int num)
    {
        for (int col = 0; col < N; col++)
            if (grid[row][col] == num)
                return true;
        return false;
    }

    bool UsedInCol(int col, int num)
    {
        for (int row = 0; row < N; row++)
            if (grid[row][col] == num)
                return true;
        return false;
    }

    bool UsedInBox(int boxStartRow, int boxStartCol, int num)
    {
        for (int row = 0; row < sqrt(N); row++)
            for (int col = 0; col < sqrt(N); col++)
                if (grid[row + boxStartRow][col + boxStartCol] == num)
                    return true;
        return false;
    }

    bool isSafe(int row, int col, int num)
    {
        return !UsedInRow(row, num) && !UsedInCol(col, num) &&
               !UsedInBox(row - row % int(sqrt(N)), col - col % int(sqrt(N)), num);
    }

    bool SolveSudoku()
    {
        int row, col;
        if (!FindUnassignedLocation(row, col))
            return true;
        for (int num = 1; num <= N; num++)
        {
            if (isSafe(row, col, num))
            {
                grid[row][col] = num;
                if (SolveSudoku())
                    return true;
                grid[row][col] = UNASSIGNED;
            }
        }
        return false;
    }

public:
    Sudoku(const string &filename)
    {
        ifstream puzzle_file(filename);
        string line;
        getline(puzzle_file, line);
        vector<int> integers;
        istringstream iss(line);
        
        int number;
        while (iss >> number)
            integers.push_back(number);
        N = integers.size();

        puzzle_file.close();
        puzzle_file.open(filename);

        // Allocate memory for the grid
        grid = new int *[N];
        for (int i = 0; i < N; ++i)
            grid[i] = new int[N];

        // Assuming the rest of the file contains the puzzle
        for (int h = 0; h < N; h++)
            for (int w = 0; w < N; w++)
                puzzle_file >> grid[h][w];
        puzzle_file.close();
    }

    ~Sudoku()
    {
        // Deallocate memory for the grid
        for (int i = 0; i < N; ++i)
            delete[] grid[i];
        delete[] grid;
    }

    void SolveAndPrint()
    {
        if (SolveSudoku())
            printGrid();
        else
            cout << "No solution exists" << endl;
    }

    void printGrid()
    {
        for (int row = 0; row < N; row++)
        {
            for (int col = 0; col < N; col++)
                cout << grid[row][col] << " ";
            cout << endl;
        }
    }
};

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Input file not found" << endl;
        return 1;
    }

    string filename = argv[1]; // Use the provided input file name from the command line
    Sudoku sudoku(filename);
    sudoku.SolveAndPrint();

    return 0;
}
