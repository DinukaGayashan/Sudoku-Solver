#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

class Sudoku
{
private:
    const int UNASSIGNED = 0;
    int N, sqrtN;
    std::vector<std::vector<int>> grid;

    bool FindUnassignedLocation(int &row, int &col)
    {
        for (row = 0; row < N; ++row)
            for (col = 0; col < N; ++col)
                if (grid[row][col] == UNASSIGNED)
                    return true;
        return false;
    }

    bool UsedInRow(int row, int num) const
    {
        for (int col = 0; col < N; ++col)
            if (grid[row][col] == num)
                return true;
        return false;
    }

    bool UsedInCol(int col, int num) const
    {
        for (int row = 0; row < N; ++row)
            if (grid[row][col] == num)
                return true;
        return false;
    }

    bool UsedInBox(int boxStartRow, int boxStartCol, int num) const
    {
        for (int row = 0; row < static_cast<int>(sqrtN); ++row)
            for (int col = 0; col < static_cast<int>(sqrtN); ++col)
                if (grid[row + boxStartRow][col + boxStartCol] == num)
                    return true;
        return false;
    }

    bool isSafe(int row, int col, int num) const
    {
        return !UsedInRow(row, num) && !UsedInCol(col, num) &&
               !UsedInBox(row - row % static_cast<int>(sqrtN), col - col % static_cast<int>(sqrtN), num);
    }

    bool SolveSudoku()
    {
        int row, col;
        if (!FindUnassignedLocation(row, col))
            return true;
        for (int num = 1; num <= N; ++num)
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
    explicit Sudoku(const std::string &filename)
    {
        std::ifstream puzzle_file(filename);
        std::string line;
        std::getline(puzzle_file, line);

        std::vector<int> integers;
        std::istringstream iss(line);

        int number;
        while (iss >> number)
            integers.push_back(number);

        N = static_cast<int>(integers.size());
        sqrtN = static_cast<int>(sqrt(N));

        puzzle_file.close();
        puzzle_file.open(filename);

        // Allocate memory for the grid
        grid.resize(N, std::vector<int>(N, 0));

        // Assuming the rest of the file contains the puzzle
        for (int h = 0; h < N; ++h)
            for (int w = 0; w < N; ++w)
                puzzle_file >> grid[h][w];

        puzzle_file.close();
    }

    ~Sudoku() = default;

    void SolveAndPrint(std::string filename)
    {
        if (SolveSudoku())
            printGrid(filename);
        else
            std::cout << "No solution exists" << std::endl;
    }

    void printGrid(std::string filename)
    {
        std::string output_filename = filename.substr(0, filename.find_last_of('.')) + "_output.txt";
        std::ofstream output_file(output_filename);
        for (int row = 0; row < N; row++)
        {
            for (int col = 0; col < N; col++)
                output_file << grid[row][col] << " ";
            output_file << std::endl;
        }
    }
};

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Input file not found" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    Sudoku sudoku(filename);
    sudoku.SolveAndPrint(filename);

    return 0;
}
