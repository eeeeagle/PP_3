#include "Matrix.hpp"

int main(int argc, char** argv)
{
	system("title Parallel Programming [Lab ¹3]");
	
	Matrix<T> a, b, c;
	std::string str[3]; 
	int taskid, threads;
	std::pair<int, int> a_size, b_size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &threads);

	if (threads < 2)
	{
		std::cout << "Not enough threads: " << threads << "\nMinimum required is 2.";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (argc != 1 && argc != 4 && (argc == 2 && strcmp(argv[1], "--help") == 0))
	{
		std::cout << "Locate paths to matrix files in arguments, to output file and specify number of threads\n\n"
			<< "EXAMPLE:\n"
			<< "    .../PP_3.exe <matrix_1_path> <matrix_2_path> <output_path>\n\n";
		MPI_Finalize();
		_exit(EXIT_FAILURE);
	}

	if (taskid == MASTER)
	{
		if (argc == 4)
		{
			str[0] = argv[1];
			str[1] = argv[2];
			str[2] = argv[3];
		}
		else
		{
			std::cout << "Locate path to matrix A: ";
			std::cin >> str[0];

			std::cout << "Locate path to matrix B: ";
			std::cin >> str[1];

			std::cout << "\nLocate path to output file: ";
			std::cin >> str[2];
		}

		std::cout << "\n\n";
		std::cout << "Reading matrix A";
		a.read_file(str[0]);
		a_size = a.size();

		std::cout << "\rReading matrix B";
		b.read_file(str[1]);
		b_size = b.size();

		std::cout << "\rPerforming C = A * B";
	}

	MPI_Bcast(&a_size.first, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&a_size.second, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

	MPI_Bcast(&b_size.first, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&b_size.second, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

	if (taskid > MASTER)
	{
		a = Matrix<T>(a_size.first, a_size.second);
		b = Matrix<T>(b_size.first, b_size.second);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	c = a * b;
	
	if (taskid == MASTER)
	{
		std::cout << "\rWriting matrix C to file [" << str[2] << "]";
		c.write_file(str[2]);
		std::cout << std::string(str[2].size() + 40, ' ');

		const std::string verificator_path = "verification.txt";
		std::cout << "\rChecking results by Python's NumPy...\r";
		system(("python verificator.py " + str[0] + ' ' + str[1] + ' ' + str[2] + " > " + verificator_path).c_str());

		std::string buffer = "False";
		try
		{
			std::ifstream file;
			file.exceptions(std::ifstream::badbit);
			file.open(verificator_path);

			getline(file, buffer);
			file.close();
			remove(verificator_path.c_str());
		}
		catch (std::ios_base::failure const& ex)
		{
			std::cout << "READING ERROR: " << ex.what() << '\n';
			MPI_Finalize();
			_exit(EXIT_FAILURE);
		}
		if (buffer == "False")
		{
			std::cout << "Matrix multiplication wasn't done correctly\n";
			MPI_Finalize();
			_exit(EXIT_FAILURE);
		}

		std::cout << "Adding multiplication results in [" << str[2] << "]...\r";
		std::cout << std::string(str[2].size() + 40, ' ');
		c.write_multiplication_result(str[2]);

		std::cout << "\rMatrix multiplication was done correctly.\n"
			"See results in [" << str[2] << "]\n\n";
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}