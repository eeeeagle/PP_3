#pragma once
#ifndef MATRIX
#define MATRIX

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <mpi.h>

#define MASTER		0          /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

bool is_master()
{
	int taskid;
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	return taskid == MASTER;
}

template<typename T>
class Matrix
{
private:
	T**		data;
	size_t  rows,
			columns;

	static double last_multiplication_time;
	
	void clear()
	{
		for (size_t i = 0; i < rows; i++)
			delete[] data[i];
		delete[] data;
	}

public:
	Matrix()
		:	data(nullptr),
			rows(0),
			columns(0)
	{}

	Matrix(const Matrix& other)
	{
		if (is_master())
		{
			rows = other.rows;
			columns = other.columns;
			data = new T * [rows];
			for (size_t i = 0; i < rows; i++)
			{
				data[i] = new T[columns];
				for (size_t j = 0; j < columns; j++)
					data[i][j] = other.data[i][j];
			}
		}
	}

	Matrix(Matrix&& other) noexcept
	{
		if (is_master())
		{
			data = other.data;
			rows = other.rows;
			columns = other.columns;

			other.data = nullptr;
			other.rows = 0;
			other.columns = 0;
		}
	}

	Matrix(const std::vector<std::vector<T>>& other)
	{
		if (is_master())
		{
			try
			{
				const size_t column_size = other.begin()->size();
				for (auto iter = other.begin() + 1; iter != other.end(); iter++)
				{
					if (iter->size() != column_size)
						throw std::logic_error("Matrix dimmension mismatch");
				}
			}
			catch (const std::logic_error& ex)
			{
				std::cout << '\n' << ex.what() << '\n';
				_exit(EXIT_FAILURE);
			}

			rows = other.size();
			columns = other.begin()->size();

			data = new T * [rows];
			for (size_t i = 0; i < rows; i++)
			{
				data[i] = new T[columns];
				for (size_t j = 0; j < columns; j++)
					data[i][j] = other[i][j];
			}
		}
	}

	Matrix(const size_t& rows, const size_t& columns)
	{
		if (is_master())
		{
			try
			{
				if (rows == 0)
					throw std::logic_error("Impossible rows count: " + std::to_string(rows));

				if (columns == 0)
					throw std::logic_error("Impossible columns count: " + std::to_string(columns));
			}
			catch (const std::logic_error& ex)
			{
				std::cout << '\n' << ex.what() << '\n';
				_exit(EXIT_FAILURE);
			}

			this->rows = rows;
			this->columns = columns;

			data = new T * [rows];
			for (size_t i = 0; i < rows; i++)
				data[i] = new T[columns];
		}
	}

	~Matrix()
	{
		clear();
	}


	Matrix& operator= (const Matrix& other)
	{
		if (this == &other)
			return *this;

		if (is_master())
		{
			T** temp_data = new T * [other.rows];
			for (size_t i = 0; i < other.rows; i++)
			{
				temp_data[i] = new T[other.columns];
				for (size_t j = 0; j < other.columns; j++)
					temp_data[i][j] = other.data[i][j];
			}

			clear();

			data = temp_data;
			rows = other.rows;
			columns = other.columns;
		}
		return *this;
	}

	Matrix& operator= (Matrix&& other)
	{
		if (this == &other)
			return *this;
		if (is_master())
		{
			clear();

			data = other.data;
			rows = other.rows;
			columns = other.columns;

			other.data = nullptr;
			other.rows = 0;
			other.columns = 0;
		}
		return *this;
	}

	Matrix operator* (Matrix& other)
	{
		try
		{
			if (columns != other.rows)
				throw std::logic_error("Multiplication is impossible: mismatch in matrix A columns and matrix B rows");
		}
		catch (const std::logic_error& ex)
		{
			std::cout << '\n' << ex.what() << '\n';
			_exit(EXIT_FAILURE);
		}

		Matrix<T> c;

		int		numtasks;
		size_t	local_rows,            /* rows of matrix A sent to each worker */
				offset;				   /* used to determine rows sent to each worker */
		MPI_Status status;

		MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

		try
		{
			if (numtasks < 2)
				throw std::runtime_error("Not enough threads: " + std::to_string(numtasks) + "\nMinimum required is 2.");
		}
		catch (const std::runtime_error& ex)
		{
			std::cout << '\n' << ex.what() << '\n';
			MPI_Abort(MPI_COMM_WORLD, 1);
			_exit(EXIT_FAILURE);
		}

		int numworkers = numtasks - 1;

		if (is_master())
		{
			c = Matrix(rows, other.columns);

			double start_time = MPI_Wtime();

			size_t averow = rows / numworkers;
			size_t extra = rows % numworkers;
			offset = 0;
			for (int dest = 1; dest <= numworkers; dest++)
			{
				local_rows = (dest <= extra) ? averow + 1 : averow;
				MPI_Send(&offset, 1, MPI_UNSIGNED_LONG_LONG, dest, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&local_rows, 1, MPI_UNSIGNED_LONG_LONG, dest, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&data[offset][0], static_cast<int>(local_rows * columns), MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&other.data, static_cast<int>(columns * other.columns), MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
				offset += local_rows;
			}

			for (int source = 1; source <= numworkers; source++)
			{
				MPI_Recv(&offset, 1, MPI_UNSIGNED_LONG_LONG, source, FROM_WORKER, MPI_COMM_WORLD, &status);
				MPI_Recv(&local_rows, 1, MPI_UNSIGNED_LONG_LONG, source, FROM_WORKER, MPI_COMM_WORLD, &status);
				MPI_Recv(&c.data[offset][0], static_cast<int>(local_rows * other.columns), MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD, &status);
			}

			double end_time = MPI_Wtime();
			last_multiplication_time = end_time - start_time;
		}
		else
		{
			MPI_Recv(&offset, 1, MPI_UNSIGNED_LONG_LONG, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
			MPI_Recv(&local_rows, 1, MPI_UNSIGNED_LONG_LONG, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
			MPI_Recv(&data, static_cast<int>(local_rows * columns), MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
			MPI_Recv(&other.data, static_cast<int>(columns * other.columns), MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

			for (size_t k = 0; k < other.columns; k++)
				for (size_t i = 0; i < local_rows; i++)
				{
					c.data[i][k] = 0;
					for (size_t j = 0; j < columns; j++)
						c.data[i][k] += data[i][j] * other.data[j][k];
				}

			MPI_Send(&offset, 1, MPI_UNSIGNED_LONG_LONG, MASTER, FROM_WORKER, MPI_COMM_WORLD);
			MPI_Send(&local_rows, 1, MPI_UNSIGNED_LONG_LONG, MASTER, FROM_WORKER, MPI_COMM_WORLD);
			MPI_Send(&c.data, static_cast<int>(local_rows * other.columns), MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
			MPI_Finalize();
		}
		return c;
	}


	void read_file(const std::string& filepath)
	{
		try
		{
			std::vector<std::vector<T>> matrix;

			std::ifstream file;
			file.exceptions(std::ifstream::badbit);
			file.open(filepath);

			for (std::string buffer; getline(file, buffer); )
			{
				std::stringstream iss(buffer);

				T number;
				std::vector<T> temp;
				while (iss >> number)
					temp.push_back(number);

				matrix.push_back(temp);
			}
			file.close();

			if (matrix.empty())
				throw std::logic_error("No matrix in file \"" + filepath + '\"');

			Matrix temp(matrix);
			*this = temp;
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
		_exit(EXIT_FAILURE);
	}

	void write_file(const std::string& filepath) const
	{
		try
		{
			std::ofstream file;
			file.exceptions(std::ofstream::badbit);
			file.open(filepath);

			for (size_t i = 0; i < rows; i++)
			{
				for (size_t j = 0; j < columns; j++)
					file << data[i][j] << ' ';
				file << '\n';
			}

			file.close();
		}
		catch (std::ios_base::failure const& ex)
		{
			std::cout << "\nWRITING ERROR: " << ex.what() << '\n';
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
			_exit(EXIT_FAILURE);
		}
	}
};

template<typename T>
double Matrix<T>::last_multiplication_time = 0.0;

#endif