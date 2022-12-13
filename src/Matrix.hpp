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
		for (unsigned i = 0; i < rows; i++)
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
		:	rows(other.rows),
			columns(other.columns)
	{
		data = new T * [rows];
		for (unsigned i = 0; i < rows; i++)
		{
			data[i] = new T[columns];
			for (unsigned j = 0; j < columns; j++)
				data[i][j] = other.data[i][j];
		}
	}

	Matrix(Matrix&& other) noexcept
	{
		data = other.data;
		other.data = nullptr;
	}

	Matrix(const std::vector<std::vector<T>>& other)
	{
		const size_t column_size = other.begin()->size();
		for (auto iter = other.begin() + 1; iter != other.end(); iter++)
		{
			if (iter->size() != column_size)
				throw std::logic_error("Matrix dimmension mismatch in" + std::string(typeid(other).name()));
		}

		rows = other.size();
		columns = column_size;

		data = new T * [rows];
		for (unsigned i = 0; i < rows; i++)
		{
			data[i] = new T[columns];
			for (unsigned j = 0; j < columns; j++)
				data[i][j] = other[i][j];
		}
	}

	Matrix(const size_t rows, const size_t columns)
	{
		if (rows == 0)
			throw std::logic_error("Impossible rows count: " + std::to_string(rows));

		if (columns == 0)
			throw std::logic_error("Impossible columns count: " + std::to_string(columns));

		this->rows = rows;
		this->columns = columns;

		data = new T * [rows];
		for (unsigned i = 0; i < rows; i++)
		{
			data[i] = new T[columns];
			for (unsigned j = 0; i < columns; i++)
				data[i][j] = NULL;
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

		T** temp_data = new T * [rows];
		for (unsigned i = 0; i < rows; i++)
		{
			temp_data[i] = new T[columns];
			for (unsigned j = 0; j < columns; j++)
				temp_data[i][j] = other.data[i][j];
		}

		clear();

		data = temp_data;
		return *this;
	}

	Matrix& operator= (Matrix&& other)
	{
		if (this == &other)
			return *this;

		clear();

		data = other.data;
		other.data = nullptr;
		return *this;
	}

	Matrix operator* (Matrix& other)
	{
		try
		{
			if (columns != other.rows)
				throw std::logic_error("Multiplication is impossible: mismatch in matrix A rows and matrix B columns");
		}
		catch (const std::logic_error& ex)
		{
			std::cout << '\n' << ex.what() << '\n';
			_exit(EXIT_FAILURE);
		}

		Matrix c(rows, other.columns);

		int	numtasks,              /* number of tasks in partition */
			taskid,                /* a task identifier */
			local_rows,            /* rows of matrix A sent to each worker */
			offset;				   /* used to determine rows sent to each worker */

		MPI_Status status;

		MPI_Init(nullptr, nullptr);
		MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
		MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
		int numworkers = numtasks - 1;

		if (taskid == MASTER)
		{
			double start_time = MPI_Wtime();

			int averow = local_rows / numworkers;
			int extra = local_rows % numworkers;
			offset = 0;
			for (int dest = 1; dest <= numworkers; dest++)
			{
				local_rows = (dest <= extra) ? averow + 1 : averow;
				MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&local_rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&data[offset][0], local_rows * columns, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&other.data, columns * other.columns, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
				offset += local_rows;
			}

			for (int source = 1; source <= numworkers; source++)
			{
				MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
				MPI_Recv(&local_rows, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, &status);
				MPI_Recv(&c.data[offset][0], local_rows * other.columns, MPI_DOUBLE, source, FROM_WORKER, MPI_COMM_WORLD, &status);
			}

			double end_time = MPI_Wtime();
			last_multiplication_time = end_time - start_time;
		}
		if (taskid > MASTER)
		{
			MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
			MPI_Recv(&local_rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
			MPI_Recv(&data, local_rows * columns, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
			MPI_Recv(&other.data, columns * other.columns, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

			for (int k = 0; k < other.columns; k++)
				for (int i = 0; i < local_rows; i++)
				{
					c.data[i][k] = 0;
					for (int j = 0; j < columns; j++)
						c.data[i][k] += data[i][j] * other.data[j][k];
				}

			MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
			MPI_Send(&local_rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
			MPI_Send(&c.data, local_rows * other.columns, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);
		}

		MPI_Finalize();
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
			{
				throw std::logic_error("No matrix in file \"" + filepath + '\"');
			}

			Matrix temp(matrix);
			*this = temp;
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

			for (unsigned i = 0; i < rows; i++)
			{
				for (unsigned j = 0; j < columns; j++)
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