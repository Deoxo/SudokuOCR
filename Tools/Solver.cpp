//
// Created by mat on 11/01/24.
//

#include "Solver.h"
#include <iostream>
#include <list>

namespace Solver
{
	struct Cell
	{
		bool possible[9]{};
		char value = 0;
	};

	Matrix* ArrayToMatrix(const char* mat, int size)
	{
		Matrix* mat2 = new Matrix(size, size, 1);

		int lim = size * size;

		for (int i = 0; i < lim; i++)
			mat2->data[i] = mat[i];

		return mat2;
	}


	char* ParseInputFile(const char* path)
	{
		// read file line by line
		FILE* file = fopen(path, "r");

		if (!file)
			throw std::runtime_error("parseInputFile : File not found !");

		// reset the file pointer
		fseek(file, 0, SEEK_SET);

		char* line = nullptr;
		size_t len = 0;
		ssize_t read;

		// create the matrix
		char* mat = new char[81];

		int delta_i = 0;
		// read the lines and fill the matrix
		for (int i = 0; i < 11; i++)
		{
			read = getline(&line, &len, file);
			if (read == -1)
				throw std::runtime_error("parseInputFile : File is not well formatted !");

			if (i == 3 || i == 7)
			{
				delta_i--;
				continue;
			}

			int delta_j = 0;

			for (int j = 0; j < 11; j++)
			{
				if (j == 3 || j == 7)
				{
					delta_j--;
					continue;
				}

				mat[(i + delta_i) * 9 + j + delta_j] = line[j] == '.' ? 0 : (line[j] - '0');
			}
		}

		fclose(file);
		return mat;
	}

	bool RecSolve(Cell board[9][9], bool rows[9][9], bool cols[9][9], bool squares[9][9], int ind = 0)
	{
		if (ind == 81)
			return true;

		const int y = ind / 9, x = ind % 9;
		Cell& cell = board[y][x];

		// Skip if already filled
		if (cell.value != 0)
			return RecSolve(board, rows, cols, squares, ind + 1);

		// Try all possible values
		for (char i = 0; i < 9; ++i)
		{
			// Skip if not possible
			const int squareIndex = (y / 3) * 3 + (x / 3);
			if (!cell.possible[i] || rows[y][i] || cols[x][i] || squares[squareIndex][i])
				continue;

			// Try value
			cell.value = i + 1;
			rows[y][i] = true;
			cols[x][i] = true;
			squares[squareIndex][i] = true;
			if (RecSolve(board, rows, cols, squares, i + 1))
				return true;

			// Reset value
			cell.value = 0;
			rows[y][i] = false;
			cols[x][i] = false;
			squares[squareIndex][i] = false;
		}

		return false;
	}

	Matrix* Solve(const Matrix& matrix)
	{
		Cell board[9][9];
		bool rows[9][9]{};
		bool cols[9][9]{};
		bool squares[9][9]{};

		// Fill board
		for (int y = 0; y < 9; ++y)
		{
			for (int x = 0; x < 9; ++x)
			{
				const char value = (char) matrix.data[y * 9 + x];
				board[y][x].value = value;

				if (value != 0)
				{
					if (rows[y][value - 1] || cols[x][value - 1] || squares[(y / 3) * 3 + (x / 3)][value - 1])
						return nullptr;
					rows[y][value - 1] = true;
					cols[x][value - 1] = true;
					squares[(y / 3) * 3 + (x / 3)][value - 1] = true;
				}
			}
		}

		// Set possible values
		for (int y = 0; y < 9; ++y)
		{
			for (int x = 0; x < 9; ++x)
			{
				const char value = board[y][x].value;
				if (value != 0)
					continue;

				int possibleCount = 0;
				for (int i = 0; i < 9; ++i)
				{
					board[y][x].possible[i] = !rows[y][i] && !cols[x][i] && !squares[(y / 3) * 3 + (x / 3)][i];
					if (board[y][x].possible[i])
						++possibleCount;
				}
				if (possibleCount == 0)
					return nullptr;
			}
		}

		Matrix* result = nullptr;
		if (RecSolve(board, rows, cols, squares))
		{
			result = Matrix::CreateSameSize(matrix);
			for (int y = 0; y < 9; ++y)
				for (int x = 0; x < 9; ++x)
					result->data[y * 9 + x] = board[y][x].value;

		}
		return result;
	}
}