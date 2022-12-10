#pragma once
#ifndef MATRIX
#define MATRIX

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <mpi.h>

typedef std::chrono::duration<double> seconds;
typedef std::chrono::high_resolution_clock Time;

template<typename T>
using Matrix = std::vector<std::vector<T>>;


template<typename T>
double multiply_matrix(const Matrix<T>&a, const Matrix<T>& b, Matrix<T>& c, const int threads_num)
/* Return value: runtime of multiply matrix*/
{
	const size_t n = a.size();
	const size_t m = a.begin()->size();
	const size_t p = b.begin()->size();

	try
	{
		if (m != b.size())
			throw std::logic_error("Multiplication is impossible: mismatch in matrix A rows and matrix B columns");
	}
	catch (const std::logic_error& ex)
	{
		std::cout << '\n' << ex.what() << '\n';
		_exit(EXIT_FAILURE);
	}

	c = Matrix<T>(n, std::vector<T>(p, 0));

	auto start_time = Time::now();
	for (int j = 0; j < p; ++j)
	{
		for (size_t k = 0; k < m; ++k)
		{
			for (size_t i = 0; i < n; ++i)
				c[i][j] += a[i][k] * b[k][j];
		}
	}
	auto end_time = Time::now();

	return seconds(end_time - start_time).count();
}

template<typename T>
void read_file(Matrix<T>& matrix, const std::string& filepath)
{
	try
	{
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

		const size_t size = matrix.begin()->size();
		for (auto iter = matrix.begin() + 1; iter != matrix.end(); iter++)
		{
			if (iter->size() != size)
				throw std::logic_error("Matrix dimmension mismatch in file \"" + filepath + '\"');
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
	_exit(EXIT_FAILURE);
}

template<typename T>
void write_file(const Matrix<T>& matrix, const std::string& filepath)
{
	try
	{
		std::ofstream file;
		file.exceptions(std::ofstream::badbit);
		file.open(filepath);

		for (auto i = matrix.begin(); i != matrix.end(); i++) /* i - iterator for rows vector*/
		{
			for (auto j = i->begin(); j != i->end(); j++) /* j - iterator for i vector */
				file << *j << ' ';
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

void add_matrix(const size_t& volume, const double& runtime, const std::string& filepath)
{
	try
	{
		std::ofstream file;
		file.exceptions(std::ofstream::badbit);
		file.open(filepath, std::ofstream::app);

		file << '\n' << "Runtime" << ' ' << runtime << ' ' << " seconds" << '\n';
		file << "Volume" << ' ' << volume;

		file.close();
	}
	catch (std::ios_base::failure const& ex)
	{
		std::cout << "\nWRITING ERROR: " << ex.what() << '\n';
		_exit(EXIT_FAILURE);
	}
}
#endif