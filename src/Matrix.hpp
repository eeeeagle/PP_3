#pragma once
#ifndef MATRIX
#define MATRIX

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>

constexpr auto MASTER = 0;
constexpr auto ROW_START_TAG = 1;
constexpr auto ROW_END_TAG = 2;
constexpr auto A_ROWS_TAG = 3;
constexpr auto C_ROWS_TAG = 4;

constexpr auto DATA_TYPE = MPI_UNSIGNED_LONG;
using T = unsigned long;

template <typename T>
class Matrix
{
private:
	std::vector<T> data;
	int rows, columns;
	static double last_multiplication_time;

	void clear()
	{
		data.clear();
		rows = 0;
		columns = 0;
	}

public:
	Matrix()
		:
		rows(0),
		columns(0),
		data()
	{}

	Matrix(const int& rows, const int& columns)
		:
		rows(rows),
		columns(columns),
		data()
	{
		try
		{
			if (rows < 1)
				throw std::logic_error("Impossible rows count: " + std::to_string(rows));

			if (columns < 1)
				throw std::logic_error("Impossible columns count: " + std::to_string(columns));
		}
		catch (const std::logic_error& ex)
		{
			std::cout << '\n' << ex.what() << '\n';
			_exit(EXIT_FAILURE);
		}

		data = std::vector<T>(this->rows * this->columns);
	}

	Matrix(const Matrix& other)
		:
		rows(other.rows),
		columns(other.columns),
		data(other.data)
	{}

	Matrix(Matrix&& other) noexcept
		:
		rows(other.rows),
		columns(other.columns),
		data(other.data)
	{
		other.data.clear();
		other.rows = 0;
		other.columns = 0;
	}

	~Matrix()
	{
		clear();
	}


	Matrix& operator= (Matrix other)
	{
		std::swap(data, other.data);
		std::swap(rows, other.rows);
		std::swap(columns, other.columns);
		return *this;
	}

	T operator() (const int& row, const int& column) const
	{
		try
		{
			if (row < 0 || row >= rows)
				throw std::logic_error("Wrong row index: " + std::to_string(rows));

			if (column < 0 || column >= columns)
				throw std::logic_error("Wrong column index: " + std::to_string(columns));
		}
		catch (const std::logic_error& ex)
		{
			std::cout << '\n' << ex.what() << '\n';
			_exit(EXIT_FAILURE);
		}

		return data[static_cast<size_t>(row * columns + column)];
	}

	T& operator() (const int& row, const int& column)
	{
		try
		{
			if (row < 0 || row >= rows)
				throw std::logic_error("Wrong row index: " + std::to_string(rows));

			if (column < 0 || column >= columns)
				throw std::logic_error("Wrong column index: " + std::to_string(columns));
		}
		catch (const std::logic_error& ex)
		{
			std::cout << '\n' << ex.what() << '\n';
			_exit(EXIT_FAILURE);
		}

		return data[static_cast<size_t>(row * columns + column)];
	}

	friend Matrix operator* (Matrix& a, Matrix& b)
	{
		int threads, taskid;
		int row_start, row_end;
		int granularity;
		double start_time = 0.0;

		MPI_Status status;
		MPI_Request request;
		Matrix c(a.columns, b.rows);

		MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
		MPI_Comm_size(MPI_COMM_WORLD, &threads);

		if (taskid == MASTER)
		{
			try
			{
				if (a.columns != b.rows)
					throw std::logic_error("Multiplication is impossible: mismatch in matrix A columns and matrix B rows");
			}
			catch (const std::logic_error& ex)
			{
				std::cout << '\n' << ex.what() << '\n';
				MPI_Finalize();
				_exit(EXIT_FAILURE);
			}

			start_time = MPI_Wtime();

			for (int i = 1; i < threads; i++) 
			{
				granularity = (a.rows / (threads - 1));
				row_start = (i - 1) * granularity;

				if (((i + 1) == threads) && ((a.rows % (threads - 1)) != 0)) 
					row_end = a.rows;
				else 
					row_end = row_start + granularity;

				MPI_Isend(&row_start, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &request);
				MPI_Isend(&row_end, 1, MPI_INT, i, ROW_START_TAG, MPI_COMM_WORLD, &request);
				MPI_Isend(&a(row_start, 0), (row_end - row_start) * a.columns , DATA_TYPE, i, A_ROWS_TAG, MPI_COMM_WORLD, &request);
			}			
		}

		MPI_Bcast(&b(0, 0), b.rows * b.columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if(taskid > MASTER)
		{
			MPI_Recv(&row_start, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(&row_end, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(&a(row_start, 0), (row_end - row_start) * a.columns, DATA_TYPE, 0, A_ROWS_TAG, MPI_COMM_WORLD, &status);

			for (int i = row_start; i < row_end; i++) 
			{
				for (int j = 0; j < b.columns; j++) 
				{
					for (int k = 0; k < b.rows; k++) 
						c(i, j) += (a(i, k) * b(k, j));
				}
			}

			MPI_Isend(&row_start, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &request);
			MPI_Isend(&row_end, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &request);
			MPI_Isend(&c(row_start, 0), (row_end - row_start) * b.columns, DATA_TYPE, 0, C_ROWS_TAG, MPI_COMM_WORLD, &request);
		}
		
		if (taskid == MASTER)
		{
			for (int i = 1; i < threads; i++) 
			{
				MPI_Recv(&row_start, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &status);
				MPI_Recv(&row_end, 1, MPI_INT, i, ROW_START_TAG, MPI_COMM_WORLD, &status);
				MPI_Recv(&c(row_start, 0), (row_end - row_start) * b.columns, DATA_TYPE, i, C_ROWS_TAG, MPI_COMM_WORLD, &status);
			}
			double end_time = MPI_Wtime();
			last_multiplication_time = end_time - start_time;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (taskid > MASTER)
			c.clear();

		return c;
	}


	void read_file(const std::string& filepath)
	{
		try
		{
			std::ifstream file;
			file.exceptions(std::ifstream::badbit);
			file.open(filepath);

			std::vector<std::vector<T>> matrix;
			for (std::string buffer; getline(file, buffer);)
			{
				std::stringstream iss(buffer);

				T value;
				std::vector<T> temp;
				while (iss >> value)
					temp.push_back(value);

				matrix.push_back(temp);
			}
			file.close();

			if (matrix.empty())
				throw std::logic_error("No matrix in file \"" + filepath + '\"');

			const size_t column_size = matrix.begin()->size();
			for (auto iter = matrix.begin() + 1; iter != matrix.end(); iter++)
			{
				if (iter->size() != column_size)
					throw std::logic_error("Matrix dimmension mismatch");
			}

			clear();

			rows = static_cast<int>(matrix.size());
			columns = static_cast<int>(column_size);
			data = std::vector<T>(rows * columns);
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < columns; j++)
					(*this)(i, j) = matrix[i][j];
			}

			return;
		}
		catch (std::ios_base::failure const& ex)
		{
			std::cout << "\nREADING ERROR: " << ex.what() << '\n';
		}
		catch (std::logic_error const& ex)
		{
			std::cout << "\nLOGIC ERROR: " << ex.what() << '\n';
		}
		MPI_Finalize();
		_exit(EXIT_FAILURE);
	}

	void write_file(const std::string& filepath) const
	{
		try
		{
			std::ofstream file;
			file.exceptions(std::ofstream::badbit);
			file.open(filepath);

			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < columns; j++)
					file << data[static_cast<size_t>(i * columns + j)] << ' ';
				file << '\n';
			}

			file.close();
		}
		catch (std::ios_base::failure const& ex)
		{
			std::cout << "\nWRITING ERROR: " << ex.what() << '\n';
			MPI_Finalize();
			_exit(EXIT_FAILURE);
		}
	}

	void write_multiplication_result(const std::string& filepath) const
	{
		try
		{
			std::ofstream file;
			file.exceptions(std::ofstream::badbit);
			file.open(filepath, std::ofstream::app);

			file << '\n' << "Runtime" << ' ' << last_multiplication_time << ' ' << " seconds" << '\n';
			file << "Volume" << ' ' << rows * columns;

			file.close();
		}
		catch (std::ios_base::failure const& ex)
		{
			std::cout << "\nWRITING ERROR: " << ex.what() << '\n';
			MPI_Finalize();
			_exit(EXIT_FAILURE);
		}
	}

	std::pair<int, int> size() const
	{
		return std::pair(rows, columns);
	}
};

double Matrix<T>::last_multiplication_time = 0.0;
#endif