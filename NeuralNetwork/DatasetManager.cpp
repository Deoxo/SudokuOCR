//
// Created by mat on 11/11/23.
//

#define _POSIX_C_SOURCE 200809L // Removes warning for getline

#include <err.h>
#include <dirent.h>
#include <sys/stat.h>
#include "DatasetManager.h"
#include "../Imagery/Imagery.h"

Matrix*** LoadMnist(const char* path, const int dataLength, const int format2D)
{
	printf("Loading MNIST...\n");
	const int cols = format2D ? 28 : 1;
	const int rows = format2D ? 28 : 784;

	// 2 arrays of Matrix* (inputs and outputs)
	Matrix*** dataset = new Matrix** [2];
	dataset[0] = new Matrix* [dataLength];
	dataset[1] = new Matrix* [dataLength];

	FILE* file = fopen(path, "r");
	if (file == nullptr)
		errx(EXIT_FAILURE, "Could not open file %s\n", path);

	char* line = nullptr;
	size_t len = 0;
	int i = 0, ind = 0;

	while (ind < dataLength && getline(&line, &len, file) != -1)
	{
		i = -1;
		dataset[0][ind] = new Matrix(rows, cols, 1);
		dataset[1][ind] = new Matrix(10, 1, 1);
		dataset[1][ind]->Reset();
		char* token = strtok(line, ",");
		while (token != nullptr)
		{
			if (i == -1)
				dataset[1][ind]->data[atoi(token)] = 1;
			else
				dataset[0][ind]->data[i] = atof(token) / 255.f;

			token = strtok(nullptr, ",");
			i++;
		}
		ind++;
	}

	printf("Done loading MNIST\n");
	return dataset;
}

Matrix*** LoadCustom(const char* path, const int dataLength, const int format2D)
{
	printf("Loading custom dataset...\n");

	const int cols = format2D ? 28 : 1;
	const int rows = format2D ? 28 : 784;

	// 2 arrays of Matrix* (inputs and outputs)
	Matrix*** dataset = new Matrix** [2];
	dataset[0] = new Matrix* [dataLength];
	dataset[1] = new Matrix* [dataLength];

	// Loop through files in the folder
	struct dirent* entry;
	DIR* dir = opendir(path);
	if (dir == nullptr)
		errx(EXIT_FAILURE, "Could not open directory %s\n", path);

	int ind = 0;
	while ((entry = readdir(dir)) != nullptr && ind < dataLength)
	{
		struct stat statbuf;
		char* filePath = (char*) malloc(sizeof(char) * (strlen(path) + strlen(entry->d_name) + 2));
		strcpy(filePath, path);
		strcat(filePath, "/");
		strcat(filePath, entry->d_name);

		if (stat(filePath, &statbuf) == -1)
		{
			perror("stat");
			delete filePath;
			continue;
		}

		if (S_ISREG(statbuf.st_mode))
		{
			dataset[0][ind] = new Matrix(rows, cols, 1);
			dataset[1][ind] = new Matrix(9, 1, 1);
			dataset[1][ind]->Reset();

			Matrix* m = Imagery::LoadImageAsMatrix(filePath);
			for (int i = 0; i < rows * cols; i++)
				dataset[0][ind]->data[i] = m->data[i] / 255.f;

			// digitIndex is written as digit_index.png, so we can get the digit by searching for digit_index_
			const int digitIndex = (int) (strstr(entry->d_name, "digit_") + 6)[0] - '0' - 1;
			dataset[1][ind]->data[digitIndex] = 1;

			delete m;
			ind++;
		}

		delete filePath;
	}

	closedir(dir);
	printf("Done loading custom dataset\n");
	return dataset;
}

Matrix*** LoadCustom3(const char* path, int dataLength, int format2D)
{
	printf("Loading custom3 dataset...\n");

	const int cols = format2D ? 28 : 1;
	const int rows = format2D ? 28 : 784;

	// 2 arrays of Matrix* (inputs and outputs)
	Matrix*** dataset = new Matrix** [2];
	dataset[0] = new Matrix* [dataLength];
	dataset[1] = new Matrix* [dataLength];

	// Loop through files in the folder
	struct dirent* entry;
	DIR* dir = opendir(path);
	if (dir == nullptr)
		errx(EXIT_FAILURE, "Could not open directory %s\n", path);

	int ind = 0;
	while ((entry = readdir(dir)) != nullptr && ind < dataLength)
	{
		struct stat statbuf;
		char* filePath = (char*) malloc(sizeof(char) * (strlen(path) + strlen(entry->d_name) + 2));
		strcpy(filePath, path);
		strcat(filePath, "/");
		strcat(filePath, entry->d_name);

		if (stat(filePath, &statbuf) == -1)
		{
			perror("stat");
			delete filePath;
			continue;
		}

		if (S_ISREG(statbuf.st_mode))
		{
			dataset[0][ind] = new Matrix(rows, cols, 1);
			dataset[1][ind] = new Matrix(10, 1, 1);
			dataset[1][ind]->Reset();

			Matrix* m = Imagery::LoadImageAsMatrix(filePath);
			for (int i = 0; i < rows * cols; i++)
				dataset[0][ind]->data[i] = m->data[i] / 255.f;

			// digitIndex is written as digit_index.png, so we can get the digit by searching for digit_index_
			const int digitIndex = (int) (entry->d_name[6] - '0' - 1);
			dataset[1][ind]->data[digitIndex] = 1;

			delete m;
			ind++;
		}

		delete filePath;
	}

	closedir(dir);
	printf("Done loading custom3 dataset\n");
	return dataset;
}
