#include "Matrix.hpp"
#include <string>
#define PATH_MAX 255

bool is_exists(const std::string& name) 
{
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

int main(int argc, char** argv)
{
	system("title Parallel Programming [Lab ¹3]");

	if (argc > 4 || (argc == 2 && std::string(argv[1]) == "--help"))
	{
		std::cout << "Locate paths to matrix files in arguments, to output file and specify number of threads\n\n"
			<< "EXAMPLE:\n"
			<< "    .../PP_3.exe <matrix_1_path> <matrix_2_path> <output_path>\n\n";
		_exit(EXIT_FAILURE);
	}

	std::string str[3];
	try
	{
		if (argc == 4)
		{
			if (!is_exists(argv[1]))
				throw std::invalid_argument(std::to_string(1) + "]: " + argv[1] + " (File doesn't exist)");
			str[0] = argv[1];

			if (!is_exists(argv[2]))
				throw std::invalid_argument(std::to_string(2) + "]: " + argv[2] + " (File doesn't exist)");
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
	}
	catch (std::invalid_argument const& ex)
	{
		std::cout << std::string("INVALID ARGUMENT [") << ex.what() << '\n'
			<< "Use [.../PP_3.exe --help] to get more info\n";
		_exit(EXIT_FAILURE);
	}

	std::cout << "\n\n";
	Matrix<unsigned long> a, b;

	std::cout << "Reading matrix A";
	a.read_file(str[0]);

	std::cout << "\rReading matrix B";
	b.read_file(str[1]);

	std::cout << "\rPerforming C = A * B";
	
	Matrix c = a * b;
	
	std::cout << "\rWriting matrix C to file [" << str[2] << "]";
	c.write_file(str[2]);

	for (size_t i = str[2].size() + 40; i > 0; i--)
		std::cout << ' ';

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
		_exit(EXIT_FAILURE);
	}
	if (buffer == "False")
	{
		std::cout << "Matrix multiplication wasn't done correctly\n";
		_exit(EXIT_FAILURE);
	}

	std::cout << "Adding multiplication results in [" << str[2] << "]...\r";

	for (size_t i = str[2].size() + 40; i > 0; i--)
		std::cout << ' ';

	c.write_multiplication_result(str[2]);

	std::cout << "\rMatrix multiplication was done correctly.\n"
				 "See results in [" << str[2] << "]\n\n";
	return 0;
}