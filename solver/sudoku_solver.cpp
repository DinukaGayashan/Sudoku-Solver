#include <cstdint>
#include <array>
#include <tuple>
#include <vector>
#include <cmath>
#include <iterator>
#include <fstream>
#include <sstream>

using namespace std;

array<uint32_t, 16> rows{}, cols{}, boxes{};
vector<tuple<int, int, int>> todo_cells;
int num_todo = 0;
bool solved = false;

int number_of_candidates(const tuple<int, int, int> &row_col_box)
{
    int row = get<0>(row_col_box);
    int col = get<1>(row_col_box);
    int box = get<2>(row_col_box);
    int candidates = rows[row] & cols[col] & boxes[box];
    return __builtin_popcount(candidates);
}

void take_best_to_front(int &todo_index)
{
    vector<tuple<int, int, int>>::iterator first = todo_cells.begin() + todo_index;
    vector<tuple<int, int, int>>::iterator best = first;
    int best_count = number_of_candidates(*best);
    for (vector<tuple<int, int, int>>::iterator next = first + 1; best_count > 1 && next < todo_cells.end(); ++next)
    {
        int next_count = number_of_candidates(*next);
        if (next_count < best_count)
        {
            best_count = next_count;
            best = next;
        }
    }
    swap(*first, *best);
}

void solve_puzzle(int todo_index, vector<int> &solution, int &puzzle_size)
{
    take_best_to_front(todo_index);

    int row = get<0>(todo_cells[todo_index]);
    int col = get<1>(todo_cells[todo_index]);
    int box = get<2>(todo_cells[todo_index]);

    int candidates = rows[row] & cols[col] & boxes[box];
    while (candidates)
    {
        uint32_t candidate = candidates & -candidates;
        rows[row] ^= candidate;
        cols[col] ^= candidate;
        boxes[box] ^= candidate;
        solution[row * puzzle_size + col] = __builtin_ffs(candidate);

        if (todo_index < num_todo)
            solve_puzzle(todo_index + 1, solution, puzzle_size);
        else
            solved = true;
        if (solved)
            return;

        rows[row] ^= candidate;
        cols[col] ^= candidate;
        boxes[box] ^= candidate;
        candidates = candidates & (candidates - 1);
    }
}

bool initialize_puzzle(const vector<int> &input, vector<int> &solution, int &puzzle_size, int &sqrt_puzzle_size, uint32_t all_possibilities)
{
    rows.fill(all_possibilities);
    cols.fill(all_possibilities);
    boxes.fill(all_possibilities);
    solution = input;

    for (int row = 0; row < puzzle_size; ++row)
    {
        for (int col = 0; col < puzzle_size; ++col)
        {
            int box = row / sqrt_puzzle_size * sqrt_puzzle_size + col / sqrt_puzzle_size;
            int value = input[row * puzzle_size + col];

            if (value != 0)
            {
                uint32_t int_value = (value - 1);
                uint32_t bit = 1u << int_value;

                if (rows[row] & bit && cols[col] & bit && boxes[box] & bit)
                {
                    rows[row] ^= bit;
                    cols[col] ^= bit;
                    boxes[box] ^= bit;
                }
                else
                    return false;
            }
            else
                todo_cells.emplace_back(make_tuple(row, col, box));
        }
    }
    num_todo = todo_cells.size() - 1;
    return true;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
        return 1;
    string filename = argv[1];

    vector<int> input;
    ifstream puzzle_file(filename);
    string line;
    getline(puzzle_file, line);
    istringstream iss(line);
    input.insert(input.end(), istream_iterator<int>(iss), istream_iterator<int>());
    int puzzle_size = static_cast<int>(input.size());
    int sqrt_puzzle_size = static_cast<int>(sqrt(puzzle_size));

    puzzle_file.close();
    puzzle_file.open(filename);
    input.clear();
    input.insert(input.end(), istream_iterator<int>(puzzle_file), istream_iterator<int>());
    puzzle_file.close();

    uint32_t all_possibilities = puzzle_size == 9 ? 0x1ff : 0xffff;
    vector<int> solution;
    string output_filename = filename.substr(0, filename.find_last_of('.')) + "_output.txt";
    ofstream output_file(output_filename);

    if (initialize_puzzle(input, solution, puzzle_size, sqrt_puzzle_size, all_possibilities))
    {
        solve_puzzle(0, solution, puzzle_size);
        for (int i = 0; i < puzzle_size; i++, output_file << "\n")
            for (int j = 0; j < puzzle_size; j++)
                output_file << solution[i * puzzle_size + j] << " ";
    }
    else
        output_file << "No Solution";
    output_file.close();

    return 0;
}
