//
// Created by mat on 27/12/23.
//

#include "Imagery.h"
#include <QImage>
#include <QDir>
#include "../Tools/Settings.h"
#include "GridDetection.h"
#include "NeuralNetwork/Network.h"
#include "Tools/FileManagement.h"

namespace Imagery
{
	Matrix* LoadImageAsMatrix(const QString& imagePath)
	{
		QImage image(imagePath);
		if (image.isNull())
			throw std::runtime_error("Failed to load image at " + imagePath.toStdString());


		int rows = image.height();
		int cols = image.width();
		int dims = 3; // Three color channels (RGB)

		Matrix* matrix = new Matrix(rows, cols, dims); // Create a Matrix object

		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				QRgb pixel = image.pixel(j, i);
				matrix->data[i * matrix->cols + j + 0 * matrix->matrixSize] =
						static_cast<float>(qRed(pixel));   // Red channel
				matrix->data[i * matrix->cols + j + 1 * matrix->matrixSize] =
						static_cast<float>(qGreen(pixel)); // Green channel
				matrix->data[i * matrix->cols + j + 2 * matrix->matrixSize] =
						static_cast<float>(qBlue(pixel));  // Blue channel
			}
		}

		return matrix;
	}

	Matrix* ConvertToGrayscale(const Matrix& m)
	{
		Matrix* gray = new Matrix(m.rows, m.cols, 1);
		for (int i = 0; i < m.matrixSize; i++)
			gray->data[i] = (float) (0.299 * m.data[i] + 0.587 * m.data[m.matrixSize + i] +
									 0.114 * m.data[2 * m.matrixSize + i]);
		return gray;
	}

	Matrix* GetGaussianKernel(const int size)
	{
		Matrix* kernel = new Matrix(size, size, 1);

		float* gauss = kernel->data;
		double sum = 0;
		int i, j;

		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				double x = i - (size - 1) / 2.0;
				double y = j - (size - 1) / 2.0;
				gauss[i * size + j] =
						GAUSS_BLUR_K * exp(((pow(x, 2) + pow(y, 2)) / ((2 * pow(GAUSS_BLUR_SIGMA, 2)))) * (-1));
				sum += gauss[i * size + j];
			}
		}
		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				gauss[i * size + j] /= (float) sum;
			}
		}

		return kernel;
	}

	void Blur(const Matrix& m, Matrix& output, const int strength)
	{
		Matrix* kernel = GetGaussianKernel(strength);
		Matrix::ValidConvolution(m, *kernel, output);

		delete kernel;
	}

	void
	AdaptiveThreshold(const Matrix& m, Matrix& output, const int neighborhoodSize, const float offset,
					  const float highValue, const float lowValue)
	{
		// Adaptive threshold with gaussian weighted sum of the neighbourhood values
		Matrix* kernel = GetGaussianKernel(neighborhoodSize);
		Matrix* thresh = Matrix::CreateSameSize(m);
		Matrix::ValidConvolution(m, *kernel, *thresh);
		for (int i = 0; i < output.matrixSize; i++)
			output.data[i] = m.data[i] > thresh->data[i] - offset ? highValue : lowValue;

		delete thresh;
		delete kernel;
	}

	void Dilate(const Matrix& m, Matrix& output, const int neighborhoodSize)
	{
		Matrix* kernel = new Matrix(neighborhoodSize, neighborhoodSize, 1);
		for (int i = 0; i < kernel->matrixSize; i++)
			kernel->data[i] = 0;
		for (int i = 0; i < neighborhoodSize; i++)
		{
			kernel->data[i * neighborhoodSize + neighborhoodSize / 2] = 1;
			kernel->data[(neighborhoodSize / 2) * neighborhoodSize + i] = 1;
		}

		const int outputCols = m.cols;
		const int outputRows = m.rows;

#if SAFE
		if (output.cols != outputCols || outputRows != output.rows)
		{
			fprintf(stderr, "%s%i%s%i%s", "right shape is : (", outputRows, ",", outputCols, ")\n");
			throw std::runtime_error("MatValidConvolution : Output Matrix has not the right shape ! ");
		}
#endif

		const int filterCols = kernel->cols;
		const int filterRows = kernel->rows;

		const int inputCols = m.cols;
		const int inputRows = m.rows;

		const int r = (filterRows - 1) / 2;
		const int c = (filterCols - 1) / 2;
		for (int i = 0; i < outputRows; i++)
		{
			for (int j = 0; j < outputCols; j++)
			{
				float maxi = 0;
				for (int k = 0; k < filterRows; k++)
				{
					for (int l = 0; l < filterCols; l++)
					{
						const int inputRow = i + k - r;
						const int inputCol = j + l - c;
						if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
						{
							const float v = m.data[inputRow * m.cols + inputCol] * kernel->data[k * kernel->cols + l];
							if (v > maxi)
								maxi = v;
						}
					}
				}
				output.data[i * output.cols + j] = maxi;
			}
		}

		delete kernel;
	}

	void Erode(const Matrix& m, Matrix& output, const int neighborhoodSize)
	{
		Matrix* kernel = new Matrix(neighborhoodSize, neighborhoodSize, 1);
		for (int i = 0; i < kernel->matrixSize; i++)
			kernel->data[i] = 0;
		for (int i = 0; i < neighborhoodSize; i++)
		{
			kernel->data[i * neighborhoodSize + neighborhoodSize / 2] = 1;
			kernel->data[(neighborhoodSize / 2) * neighborhoodSize + i] = 1;
		}

		const int outputCols = m.cols;
		const int outputRows = m.rows;

#if SAFE
		if (output.cols != outputCols || outputRows != output.rows)
		{
			fprintf(stderr, "%s%i%s%i%s", "right shape is : (", outputRows, ",", outputCols, ")\n");
			throw std::runtime_error("MatValidConvolution : Output Matrix has not the right shape ! ");
		}
#endif

		const int filterCols = kernel->cols;
		const int filterRows = kernel->rows;

		const int inputCols = m.cols;
		const int inputRows = m.rows;

		const int r = (filterRows - 1) / 2;
		const int c = (filterCols - 1) / 2;
		for (int i = 0; i < outputRows; i++)
		{
			for (int j = 0; j < outputCols; j++)
			{
				float mini = 255;
				for (int k = 0; k < filterRows; k++)
				{
					for (int l = 0; l < filterCols; l++)
					{
						const int inputRow = i + k - r;
						const int inputCol = j + l - c;
						if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
						{
							const float ke = kernel->data[k * kernel->cols + l];
							const float v = m.data[inputRow * m.cols + inputCol] * ke;
							if (ke != 0 && v < mini)
								mini = v;
						}
					}
				}
				output.data[i * output.cols + j] = mini;
			}
		}

		delete kernel;
	}

	void BitwiseNot(const Matrix& m, Matrix& output)
	{
		for (int i = 0; i < m.matrixSize; i++)
			output.data[i] = 255 - m.data[i];
	}

	void Canny(const Matrix& m, Matrix& output, const float lowThreshold, const float highThreshold)
	{
		// Canny edge detection algorithm
		Matrix* blurred = Matrix::CreateSameSize(m);
		Blur(m, *blurred, 5);
		Matrix** gradients = GetGradients(*blurred);
		Matrix* nonMaxSuppressed = Matrix::CreateSameSize(m);
		NonMaximumSuppression(*gradients[0], *gradients[1], *nonMaxSuppressed);
		Matrix* doubleThresholded = Matrix::CreateSameSize(m);
		DoubleThreshold(*nonMaxSuppressed, *doubleThresholded, lowThreshold, highThreshold);
		Matrix& hysteresis = output;
		Hysteresis(*doubleThresholded, hysteresis, highThreshold);

		// Free the memory
		delete blurred;
		delete gradients[0];
		delete gradients[1];
		delete nonMaxSuppressed;
		delete doubleThresholded;
		delete[] gradients;
	}

	Matrix** GetGradients(const Matrix& m)
	{
		// Build the kernels
		Matrix* horizontalKernel = new Matrix(3, 3, 1);
		Matrix* verticalKernel = new Matrix(3, 3, 1);
		const float horizontalKernelData[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
		const float verticalKernelData[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
		memcpy(horizontalKernel->data, horizontalKernelData, sizeof(float) * 9);
		memcpy(verticalKernel->data, verticalKernelData, sizeof(float) * 9);

		// Convolve the kernels with the matrix
		Matrix* ix = Matrix::CreateSameSize(m);
		Matrix* iy = Matrix::CreateSameSize(m);
		Matrix::ValidConvolution(m, *horizontalKernel, *ix);
		Matrix::ValidConvolution(m, *verticalKernel, *iy);

		// Create the gradients array
		Matrix** gradients = new Matrix* [2];
		gradients[0] = ix;
		gradients[1] = iy;

		// G
		for (int i = 0; i < ix->matrixSize; i++)
			ix->data[i] = (float) sqrt(pow(ix->data[i], 2) + pow(iy->data[i], 2));

		// Theta
		for (int i = 0; i < ix->matrixSize; i++)
			iy->data[i] = std::atan2(iy->data[i], ix->data[i]);

		// Free the kernels
		delete horizontalKernel;
		delete verticalKernel;

		return gradients;
	}

	void NonMaximumSuppression(const Matrix& m, const Matrix& angles, Matrix& output)
	{
		// Convert angles to degrees in [0, 180[
		for (int i = 0; i < angles.matrixSize; i++)
		{
			angles.data[i] = (float) (angles.data[i] * 180 / M_PI);
			if (angles.data[i] < 0)
				angles.data[i] += 180;
		}

		// Non maximum suppression
		for (int x = 1; x < m.rows - 1; x++)
		{
			for (int y = 1; y < m.cols - 1; y++)
			{
				const float angle = angles.data[x * angles.cols + y];
				const float v = m.data[x * m.cols + y];
				float v1, v2;
				if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180))
				{
					v1 = m.data[x * m.cols + y - 1];
					v2 = m.data[x * m.cols + y + 1];
				}
				else if (angle >= 22.5 && angle < 67.5)
				{
					v1 = m.data[(x - 1) * m.cols + y + 1];
					v2 = m.data[(x + 1) * m.cols + y - 1];
				}
				else if (angle >= 67.5 && angle < 112.5)
				{
					v1 = m.data[(x - 1) * m.cols + y];
					v2 = m.data[(x + 1) * m.cols + y];
				}
				else
				{
					v1 = m.data[(x - 1) * m.cols + y - 1];
					v2 = m.data[(x + 1) * m.cols + y + 1];
				}
				if (v < v1 || v < v2)
					output.data[x * output.cols + y] = 0;
				else
					output.data[x * output.cols + y] = v;
			}
		}
	}

	void DoubleThreshold(const Matrix& m, Matrix& output, const float lowThreshold, const float highThreshold)
	{
		for (int i = 0; i < m.matrixSize; i++)
		{
			const float v = m.data[i];
			if (v < lowThreshold)
				output.data[i] = 0;
			else if (v > highThreshold)
				output.data[i] = 255;
			else
				output.data[i] = 127;
		}
	}

	void Hysteresis(const Matrix& m, Matrix& output, const float highThreshold)
	{
		for (int i = 0; i < m.matrixSize; i++)
		{
			const float v = m.data[i];
			if (v == 127)
			{
				const int x = i / m.cols;
				const int y = i % m.cols;
				for (int k = -1; k <= 1; k++)
				{
					for (int l = -1; l <= 1; l++)
					{
						const int inputRow = x + k;
						const int inputCol = y + l;
						if (inputRow >= 0 && inputRow < m.rows && inputCol >= 0 && inputCol < m.cols)
						{
							const float v2 = m.data[inputRow * m.cols + inputCol];
							if (v2 > highThreshold)
								output.data[x * output.cols + y] = 255;
						}
					}
				}
			}
			else
				output.data[(i / m.cols) * output.cols + i % m.cols] = v;
		}
	}

	Matrix* HoughTransform(const Matrix& m)
	{
		// Build the accumulator
		const int diagonal = (int) (sqrt(pow(m.rows, 2) + pow(m.cols, 2)) + 1);
		const int accumulatorRows = 2 * diagonal;
		const int accumulatorCols = 180;
		Matrix* accumulator = new Matrix(accumulatorRows, accumulatorCols, 1);
		accumulator->Reset();

		// Precompute cos and sin values
		float* cosTable = new float[180];
		float* sinTable = new float[180];
		for (int i = 0; i < 180; i++)
		{
			cosTable[i] = std::cos((float) i * (float) M_PI / 180);
			sinTable[i] = std::sin((float) i * (float) M_PI / 180);
		}

		// Fill the accumulator
		for (int x = 0; x < m.rows; x++)
		{
			for (int y = 0; y < m.cols; y++)
			{
				const float v = m.data[x * m.cols + y];
				if (v == 255)
				{
					for (int theta = 0; theta < 180; theta++)
					{
						int rho = (int) std::round(y * cosTable[theta] + x * sinTable[theta]) + diagonal;
						accumulator->data[rho * accumulator->cols + theta] += 1;
					}
				}
			}
		}

		// Draw hough space in a surface and save it to a file
		/*SDL_Surface* surface = SDL_CreateRGBSurface(0, accumulatorCols, accumulatorRows, 32, 0, 0, 0, 0);
		for (int rho = 0; rho < accumulatorRows; rho++)
		{
			for (int theta = 0; theta < accumulatorCols; theta++)
			{
				const int v = (int)MatGetValueAt(accumulator, rho, theta);
				Uint32 color = SDL_MapRGB(surface->format, v, v, v);
				((Uint32*)surface->pixels)[rho * accumulatorCols + theta] = color;
			}
		}
		SDL_SaveBMP(surface, "hough.bmp");
		SDL_FreeSurface(surface);*/

		// Free the memory
		delete[] cosTable;
		delete[] sinTable;

		return accumulator;
	}

	float* kMeansClustersCenter(const int* data, const int dataLength)
	{
		int* cluster1 = new int[dataLength];
		int* cluster2 = new int[dataLength];
		float* clusterCenters = new float[2];
		clusterCenters[0] = 33;
		clusterCenters[0] = -33;
		const int maxIter = 100;
		int iters = 0;
		int changed = 1;

		/*for (int i = 0; i < dataLength; i++)
			printf("angle: %i\n", data[i]);*/

		while (changed && iters < maxIter)
		{
			iters++;
			changed = 0;
			int cluster1Length = 0;
			int cluster2Length = 0;
			for (int i = 0; i < dataLength; i++)
			{
				const int v = data[i];
				const float d1 = std::abs(v - clusterCenters[0]);
				const float d2 = std::abs(v - clusterCenters[1]);
				if (d1 < d2)
				{
					cluster1[cluster1Length] = v;
					cluster1Length++;
				}
				else
				{
					cluster2[cluster2Length] = v;
					cluster2Length++;
				}
			}
			int sum1 = 0;
			int sum2 = 0;
			for (int i = 0; i < cluster1Length; i++)
				sum1 += cluster1[i];
			for (int i = 0; i < cluster2Length; i++)
				sum2 += cluster2[i];
			if (cluster1Length == 0)
			{
				clusterCenters[1] = sum2 / cluster2Length;
				clusterCenters[0] = clusterCenters[1];
				changed = 1;
			}
			else if (cluster2Length == 0)
			{
				clusterCenters[0] = sum1 / cluster1Length;
				clusterCenters[1] = clusterCenters[0];
				changed = 1;
			}
			else
			{
				const float newCenter1 = sum1 / cluster1Length;
				const float newCenter2 = sum2 / cluster2Length;
				if (newCenter1 != clusterCenters[0] || newCenter2 != clusterCenters[1])
				{
					changed = 1;
					clusterCenters[0] = newCenter1;
					clusterCenters[1] = newCenter2;
				}
			}

			/* printf("c1: %f c2: %f\n", clusterCenters[0], clusterCenters[1]);
			 printf("c1: %i c2: %i\n", cluster1Length, cluster2Length);*/
		}

		// print clusters

		for (int i = 0; i < dataLength; i++)
		{
			const int v = data[i];
			const float d1 = std::abs(v - clusterCenters[0]);
			const float d2 = std::abs(v - clusterCenters[1]);
			if (d1 < d2)
				printf("1 ");
			else
				printf("2 ");
			printf("%i\n", v);
		}

		delete[] cluster1;
		delete[] cluster2;

		return clusterCenters;
	}

	float ComputeImageAngle(const HoughLine* lines, const int numLines)
	{
		int* angles = new int[numLines];
		float avg = 0, avg2 = 0, avg2n = 0;
		for (int i = 0; i < numLines; i++)
		{
			// Skip weird lines
			if (lines[i].theta == INFINITY || lines[i].theta < 0 || lines[i].theta > 180)
				continue;
			angles[i] = (int) (lines[i].theta * 180 / M_PI); // convert to degrees
			angles[i] %= 90; // keep only the angle in [0, 90[1
			if (angles[i] == 0) // 0 angles mess up the algorithm, so we set them to 90
				angles[i] = 90;
			avg += angles[i];
		}

		avg /= (float) numLines;
		for (int i = 0; i < numLines; ++i)
			if (std::abs(angles[i] - avg) < 15)
			{
				avg2 += angles[i];
				avg2n++;
			}
		avg2 /= avg2n;

		//printf("avg: %f\n", avg);
		//printf("avg2: %f\n", avg2);

		delete[] angles;
		return avg2 > 45 ? -(90 - avg2) : avg2;

		/*float* clusterCenters = kMeansClustersCenter(angles, numLines);
		printf("clusters angles: %f %f\n", clusterCenters[0], clusterCenters[1]);

		int angle1 = abs((int) clusterCenters[0] - 90);
		int angle2 = abs((int) clusterCenters[1] - 90);
		int angle3 = abs((int) clusterCenters[0] - 0);
		int angle4 = abs((int) clusterCenters[1] - 0);
		// take min angle
		int rotationAngle = angle1;
		if (angle2 < rotationAngle)
			rotationAngle = angle2;
		if (angle3 < rotationAngle)
			rotationAngle = angle3;
		if (angle4 < rotationAngle)
			rotationAngle = angle4;

		return rotationAngle;*/
	}

	void BilateralFilter(const Matrix& input, Matrix& output, const int diameter, const float sigma_color,
						 const float sigma_space)
	{
		const int width = input.cols;
		const int height = input.rows;
		const int radius = diameter / 2;

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float sum = 0.f;
				float sum_weight = 0.f;

				for (int j = -radius; j <= radius; j++)
				{
					for (int i = -radius; i <= radius; i++)
					{
						const int nx = x + i;
						const int ny = y + j;

						if (nx >= 0 && nx < width && ny >= 0 && ny < height)
						{
							const float current = input.data[y * width + x];
							const float neighbor = input.data[ny * width + nx];

							const float color_distance = (float) abs((int) (current - neighbor));

							const float spatial_distance = sqrtf(powf(i, 2.f) + powf(j, 2.f));
							const float weight = expf(-color_distance / (2 * sigma_color * sigma_color) -
													  spatial_distance / (2 * sigma_space * sigma_space));

							sum += weight * neighbor;
							sum_weight += weight;
						}
					}
				}

				float filtered_pixel = (sum / sum_weight);

				output.data[y * width + x] = filtered_pixel;
			}
		}
	}

	void SobelEdgeDetector(const Matrix& m, Matrix& output)
	{
		const float sobel_x[3][3] = {{-1, 0, 1},
									 {-2, 0, 2},
									 {-1, 0, 1}};
		const float sobel_y[3][3] = {{-1, -2, -1},
									 {0,  0,  0},
									 {1,  2,  1}};

		const int width = m.cols;
		const int height = m.rows;

		for (int y = 1; y < height - 1; y++)
		{
			for (int x = 1; x < width - 1; x++)
			{
				float gx = 0, gy = 0;
				for (int j = -1; j <= 1; j++)
				{
					for (int i = -1; i <= 1; i++)
					{
						gx += m.data[(y + j) * width + (x + i)] * sobel_x[j + 1][i + 1];
						gy += m.data[(y + j) * width + (x + i)] * sobel_y[j + 1][i + 1];
					}
				}
				output.data[y * width + x] = sqrtf(gx * gx + gy * gy);
			}
		}
	}

	void
	AdaptiveThreshold2(const Matrix& m, Matrix& output, const float threshold, const float highValue,
					   const float lowValue)
	{
		const int width = m.cols;
		const int height = m.rows;

		const int s2 = std::max(width, height) / 16;
		float* integralImg = new float[width * height]();
		float sum = 0;
		int count = 0;
		int x1, y1, x2, y2;

		for (int y = 0; y < height; y++)
		{
			sum += m.data[0 * m.cols + y];
			integralImg[y] = sum;
		}

		for (int i = 1; i < width; i++)
		{
			sum = 0;
			for (int j = 0; j < height; j++)
			{
				sum += m.data[j * m.cols + i];
				integralImg[i * height + j] = integralImg[(i - 1) * height + j] + sum;
			}
		}

		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				x1 = std::max(i - s2, 1);
				x2 = std::min(i + s2, width - 1);
				y1 = std::max(j - s2, 1);
				y2 = std::min(j + s2, height - 1);
				count = (x2 - x1) * (y2 - y1);

				sum = integralImg[x2 * height + y2]
					  - integralImg[x2 * height + (y1 - 1)]
					  - integralImg[(x1 - 1) * height + y2]
					  + integralImg[(x1 - 1) * height + (y1 - 1)];

				// printf("Previous : %u\n", image->pixels[i][j].r);
				if (m.data[j * m.cols + i] * count < sum * (1.0 - threshold))
					output.data[j * output.cols + i] = lowValue;

				else
					output.data[j * output.cols + i] = highValue;

				// printf("After : %u\n", image->pixels[i][j].r);
			}
		}
		// Free
		delete[] integralImg;

		// Prevents from having a square on borders when the image is inverted
		for (int x = 0; x < width; ++x)
		{
			output.data[x] = 255;
			output.data[(height - 1) * output.cols + x] = 255;
		}
		for (int y = 0; y < height; ++y)
		{
			output.data[y * output.cols + 0] = 255;
			output.data[y * output.cols + width - 1] = 255;
		}
	}

	float ComputeEntropy(const Matrix& m)
	{
		float entropy = 0;
		float* histogram = new float[256]();
		for (int i = 0; i < m.matrixSize; i++)
			histogram[(int) m.data[i]]++;

		for (int i = 0; i < 256; i++)
		{
			histogram[i] /= (float) m.matrixSize;
			if (histogram[i] > 0)
				entropy += histogram[i] * log2f(histogram[i]);
		}
		entropy *= -1;

		delete[] histogram;

		return entropy;
	}

	float SegmentBlackPercentage(const Matrix& img, const Point* p1, const Point* p2)
	{
		const float dist = (p1->x - p2->x) * (p1->x - p2->x) + (p1->y - p2->y) * (p1->y - p2->y);
		if (dist == 0)
			return 0;
		const float vx = (p2->x - p1->x) / dist;
		const float vy = (p2->y - p1->y) / dist;

		int numBlackPixels = 0;
		for (int i = 1; i < SEGMENT_BLACK_PERCENTAGE_NUM_SAMPLES; ++i)
		{
			const int px = p1->x + vx * (dist / i);
			const int py = p1->y + vy * (dist / i);
			if (img.data[py * img.cols + px] == 255)
				numBlackPixels++;
		}

		return (float) numBlackPixels / (float) SEGMENT_BLACK_PERCENTAGE_NUM_SAMPLES;
	}

	void
	PrintPointScreenCoordinates(const Point* point, const int screenWidth, const int screenHeight, const int imgWidth,
								const int imgHeight)
	{
		float xRatio = (float) screenWidth / (float) imgWidth;
		float yRatio = (float) screenHeight / (float) imgHeight;

		printf("%i %i\n", (int) (point->x * xRatio), (int) (point->y * yRatio));
	}

	float StandardDeviation(const Matrix& m, const int blockSize)
	{
		const int width = m.cols;
		const int height = m.rows;
		const int radius = blockSize / 2;

		float sum = 0.f;
		float sum2 = 0.f;
		int count = 0;

		float res = 0.f;

		for (int y = radius; y < height - radius; y++)
		{
			float rr = 0;
			for (int x = radius; x < width - radius; x++)
			{
				sum = 0.f;
				sum2 = 0.f;
				count = 0;
				for (int j = -radius; j <= radius; j++)
				{
					for (int i = -radius; i <= radius; i++)
					{
						const float v = m.data[(y + j) * m.cols + x + i];
						sum += v;
						sum2 += v * v;
						count++;
					}
				}
				const float mean = sum / count;
				const float variance = sum2 / count - mean * mean;
				if (variance <= 0)
					continue;
				const float stdDev = sqrtf(variance);
				rr += stdDev;
			}
			rr /= width - 2 * radius;
			res += rr;
		}

		return res;
	}

	float Brightness(const Matrix& img)
	{
		float res = 0;
		for (int i = 0; i < img.matrixSize; ++i)
			res += img.data[i] / 255.f;

		return res / (float) img.matrixSize;
	}

	float Dispersion(const Matrix& in)
	{
		float res = 0;
		for (int x = 1; x < in.cols - 1; ++x)
		{
			for (int y = 1; y < in.rows - 1; ++y)
			{
				const float p = in.data[y * in.cols + x];
				for (int j = -1; j <= 1; ++j)
				{
					for (int k = -1; k <= 1; ++k)
					{
						int fx = x + j;
						int fy = y + k;

						const float diff = std::abs(in.data[fy * in.cols + fx] - p);
						if (diff < 128)
							res += diff;
					}
				}
			}
		}

		return res / 100000;
	}

	Matrix** CropBorders(const Matrix** cells, const int cropPercentage)
	{
		const int numCells = 81;
		const int cellSize = cells[0]->cols < cells[0]->rows ? cells[0]->cols : cells[0]->rows;
		const int cropSize = (int) (cellSize * cropPercentage / 100.f);
		const int croppedCellSize = cellSize - 2 * cropSize;
		Matrix** croppedCells = new Matrix* [numCells];
		for (int i = 0; i < numCells; ++i)
			croppedCells[i] = new Matrix(croppedCellSize, croppedCellSize, 1);

		for (int i = 0; i < numCells; ++i)
		{
			for (int x = cropSize; x < cellSize - cropSize; ++x)
			{
				for (int y = cropSize; y < cellSize - cropSize; ++y)
				{
					const float v = cells[i]->data[y * cells[i]->cols + x];
					croppedCells[i]->data[(y - cropSize) * croppedCells[i]->cols + x - cropSize] = v;
				}
			}
		}

		return croppedCells;
	}

	void RemoveBorderArtifacts(Matrix** cells)
	{
		const float whiteThreshold = .25f;
		const int d = 28 * BORDER_CROP_PERCENTAGE / 100.f;
		const int n = 28 - 2 * d;
		const float t = n * whiteThreshold;

		for (int i = 0; i < 81; ++i)
		{
			for (int j = 0; j < d; ++j)
			{
				float sum = 0, sum2 = 0, sum3 = 0, sum4 = 0;
				for (int k = d; k < 28 - d; ++k)
				{
					sum += cells[i]->data[j * cells[i]->cols + k];
					sum2 += cells[i]->data[(27 - j) * cells[i]->cols + k];
					sum3 += cells[i]->data[k * cells[i]->cols + j];
					sum4 += cells[i]->data[k * cells[i]->cols + 27 - j];
				}

				if (sum / 255.f > t)
				{
					for (int x = 0; x < 28; ++x)
						cells[i]->data[j * cells[i]->cols + x] = 0;
				}
				if (sum2 / 255.f > t)
				{
					for (int x = 0; x < 28; ++x)
						cells[i]->data[(27 - j) * cells[i]->cols + x] = 0;
				}
				if (sum3 / 255.f > t)
				{
					for (int x = 0; x < 28; ++x)
						cells[i]->data[x * cells[i]->cols + j] = 0;
				}
				if (sum4 / 255.f > t)
				{
					for (int x = 0; x < 28; ++x)
						cells[i]->data[x * cells[i]->cols + 27 - j] = 0;
				}
			}
		}
	}

	int* GetEmptyCells(const Matrix** cells, const float emptinessThreshold)
	{
		int* emptyCells = new int[81];
		for (int i = 0; i < 81; ++i)
		{
			const float sum = cells[i]->Sum();

			emptyCells[i] = sum < emptinessThreshold * cells[i]->matrixSize * 255;
		}

		return emptyCells;
	}

	Matrix** ResizeCellsTo28x28(const Matrix** cells)
	{
		const int numCells = 81;
		const int cellSize = cells[0]->cols;
		const int resizedCellSize = 28;
		Matrix** resizedCells = new Matrix* [numCells];
		for (int i = 0; i < numCells; ++i)
			resizedCells[i] = new Matrix(resizedCellSize, resizedCellSize, 1);

		for (int i = 0; i < numCells; ++i)
		{
			for (int x = 0; x < resizedCellSize; ++x)
			{
				for (int y = 0; y < resizedCellSize; ++y)
				{
					const int r = (int) ((float) y * (float) cellSize / (float) resizedCellSize);
					const int c = (int) ((float) x * (float) cellSize / (float) resizedCellSize);
					const float v = cells[i]->data[r * cells[i]->cols + c];
					resizedCells[i]->data[y * resizedCells[i]->cols + x] = v;
				}
			}
		}

		return resizedCells;
	}

// Offset pixels horizontally to center the digit. Discards pixels that go out of bounds. Offset can be negative.
	void HorizontalOffset(Matrix& m, const int offset)
	{
		if (offset == 0)
			return;

		const int width = m.cols;
		const int height = m.rows;

		if (offset > 0)
		{
			for (int y = 0; y < height; ++y)
			{
				for (int x = width - 1; x >= offset; --x)
					m.data[y * m.cols + x] = m.data[y * m.cols + x - offset];
				for (int x = 0; x < offset; ++x)
					m.data[y * m.cols + x] = 0;
			}
		}
		else
		{
			for (int y = 0; y < height; ++y)
			{
				for (int x = 0; x < width + offset; ++x)
					m.data[y * m.cols + x] = m.data[y * m.cols + x - offset];
				for (int x = width + offset; x < width; ++x)
					m.data[y * m.cols + x] = 0;
			}
		}
	}

	void VerticalOffset(Matrix& m, const int offset)
	{
		if (offset == 0)
			return;

		const int width = m.cols;
		const int height = m.rows;

		if (offset > 0)
		{
			for (int x = 0; x < width; ++x)
			{
				for (int y = height - 1; y >= offset; --y)
					m.data[y * m.cols + x] = m.data[(y - offset) * m.cols + x];
				for (int y = 0; y < offset; ++y)
					m.data[y * m.cols + x] = 0;
			}
		}
		else
		{
			for (int x = 0; x < width; ++x)
			{
				for (int y = 0; y < height + offset; ++y)
					m.data[y * m.cols + x] = m.data[(y - offset) * m.cols + x];
				for (int y = height + offset; y < height; ++y)
					m.data[y * m.cols + x] = 0;
			}
		}
	}

	Matrix** CenterCells(const Matrix** cells, const int* emptyCells)
	{
		Matrix** centeredCells = new Matrix* [81];
		for (int i = 0; i < 81; ++i)
		{
			if (emptyCells[i])
			{
				centeredCells[i] = new Matrix(28, 28, 1);
				cells[i]->CopyValuesTo(*centeredCells[i]);
				continue;
			}

			// Compute the center of pixels, which is the colum and row at the center at the digit
			const int width = cells[i]->cols;
			const int height = cells[i]->rows;
			int centerCol = 0, centerRow = 0;
			int numPixels = 0;
			for (int x = 0; x < width; ++x)
			{
				for (int y = 0; y < height; ++y)
				{
					const float v = cells[i]->data[y * cells[i]->cols + x];
					if (v > 0)
					{
						centerCol += x;
						centerRow += y;
						numPixels++;
					}
				}
			}
			centerCol /= numPixels;
			centerRow /= numPixels;

			// Compute the offset to center the digit
			const int offsetCol = 14 - centerCol;
			const int offsetRow = 14 - centerRow;

			centeredCells[i] = new Matrix(28, 28, 1);
			cells[i]->CopyValuesTo(*centeredCells[i]);
			HorizontalOffset(*centeredCells[i], offsetCol);
			VerticalOffset(*centeredCells[i], offsetRow);
		}

		return centeredCells;
	}

// Delete a pixel and recursively delete its neighbors if they are white.
	void RemoveContinuousPixels(Matrix& img, const int y, const int x) // NOLINT(*-no-recursion)
	{
		if (y < 0 || y >= img.rows || x < 0 || x >= img.cols)
			return;
		if (img.data[y * img.cols + x] == 0)
			return;

		img.data[y * img.cols + x] = 0;

		RemoveContinuousPixels(img, y - 1, x);
		RemoveContinuousPixels(img, y + 1, x);
		RemoveContinuousPixels(img, y, x - 1);
		RemoveContinuousPixels(img, y, x + 1);
	}

	void RemoveLines(Matrix& img)
	{
		for (int x = 0; x < img.cols; ++x)
		{
			RemoveContinuousPixels(img, 0, x);
			RemoveContinuousPixels(img, img.rows - 1, x);
		}
		for (int y = 0; y < img.rows; ++y)
		{
			RemoveContinuousPixels(img, y, 0);
			RemoveContinuousPixels(img, y, img.cols - 1);
		}
	}

	Matrix* Rotation(const Matrix& matrix, const Square& s, const double degree)
	{
		double angle = (-1) * degree * (M_PI / 180);
		double cos_val = cos(angle);
		double sin_val = sin(angle);

		double center_x = (s.topLeft.x + s.topRight.x + s.bottomLeft.x + s.bottomRight.x) / 4.;
		double center_y = (s.topLeft.y + s.topRight.y + s.bottomLeft.y + s.bottomRight.y) / 4.;
		//double center_x = matrix->cols / 2;
		//double center_y = matrix->rows / 2;

		Matrix* res = new Matrix(matrix.rows, matrix.cols, 1);
		for (int i = 0; i < matrix.rows; i++)
			for (int j = 0; j < matrix.cols; j++)
				res->data[i * res->cols + j] = 0;

		/*double dx = sqrt(pow(s.topRight.x - s.topLeft.x, 2) + pow(s.topRight.x - s.topLeft.x, 2));
		double dy = sqrt(pow(s.bottomLeft.x - s.topLeft.x, 2) + pow(s.bottomLeft.x - s.topLeft.x, 2));
		double du = (dx + dy) / 2;

		int sx = (s.topLeft.y - center_x) * cos_val - (s.topLeft.x - center_y) * sin_val + center_x;
		int sy = (s.topLeft.y - center_x) * sin_val + (s.topLeft.x - center_y) * cos_val + center_y;*/
		//printf("%f\n", sx);
		//printf("%f\n", sy);

		/*for (int x = 0; x < dx; x++)
		{
			for (int y = 0; y < dy; y++)
			{
				int mx = x + sx;
				int my = y + sy;
				double new_x = (my - center_x) * cos_val - (mx - center_y) * sin_val + center_x;
				double new_y = (my - center_x) * sin_val + (mx - center_y) * cos_val + center_y;

				if (0 <= new_x && new_x < matrix->cols && 0 <= new_y && new_y < matrix->rows)
					MatSetValueAt(res, x, y, MatGetValueAt(matrix, (int)new_y, (int)new_x));
			}
		}*/

		for (int i = 0; i < matrix.rows; i++)
		{
			for (int j = 0; j < matrix.cols; j++)
			{
				double x = j - center_x;
				double y = i - center_y;

				double new_x = x * cos_val - y * sin_val + center_x;
				double new_y = x * sin_val + y * cos_val + center_y;

				if (0 <= new_x && new_x < matrix.cols && 0 <= new_y && new_y < matrix.rows)
					res->data[i * res->cols + j] = matrix.data[((int) new_y) * matrix.cols + (int) new_x];
			}
		}

		/*printf("hi\n");
		Matrix* realRes = new Matrix((int)dx, (int)dy);
		for (int i = 0; i < (int)dx; i++)
			for (int j = 0; j < (int)dy; j++)
				MatSetValueAt(realRes, i, j, MatGetValueAt(res, sx + j, sy + i));
		printf("bye\n");*/
		return res;
	}

	void RotatePoint(const Point& pt, const Point& center, const float angle, Point& res)
	{
		const float radians = angle * (float) M_PI / 180.0f;
		const float ptxf = static_cast<float>(pt.x);
		const float ptyf = static_cast<float>(pt.y);
		const float ctxf = static_cast<float>(center.x);
		const float ctyf = static_cast<float>(center.y);
		const int rotated_x = (int) ((ptxf - ctxf) * std::cos(radians) - (ptyf - ctyf) * std::sin(radians) + ctxf);
		const int rotated_y = (int) ((ptxf - ctxf) * std::sin(radians) + (ptyf - ctyf) * std::cos(radians) + ctyf);

		res.x = rotated_x;
		res.y = rotated_y;
	}

	Square* RotateSquare(const Square& s, const Point& center, const float angle)
	{
		Square* res = new Square();
		RotatePoint(s.topLeft, center, angle, res->topLeft);
		RotatePoint(s.topRight, center, angle, res->topRight);
		RotatePoint(s.bottomLeft, center, angle, res->bottomLeft);
		RotatePoint(s.bottomRight, center, angle, res->bottomRight);
		return res;
	}

	float Dist(const Point& pt1, const Point& pt2)
	{
		return std::sqrt(
				std::pow<float, float>(pt1.x - pt2.x, 2) + powf(pt1.y - pt2.y, 2)); // NOLINT(*-narrowing-conversions)
	}

	Point* ClosestEdgeFrom(const Square& s, const Point& pt)
	{
		Point* res = new Point();
		float minDist = Dist(s.topLeft, pt);
		res->x = s.topLeft.x;
		res->y = s.topLeft.y;
		float dist = Dist(s.topRight, pt);
		if (dist < minDist)
		{
			minDist = dist;
			res->x = s.topRight.x;
			res->y = s.topRight.y;
		}
		dist = Dist(s.bottomLeft, pt);
		if (dist < minDist)
		{
			minDist = dist;
			res->x = s.bottomLeft.x;
			res->y = s.bottomLeft.y;
		}
		dist = Dist(s.bottomRight, pt);
		if (dist < minDist)
		{
			res->x = s.bottomRight.x;
			res->y = s.bottomRight.y;
		}
		return res;
	}

	Square* Order(const Square& s, const int w, const int h)
	{
		Square* res = (Square*) malloc(sizeof(Square));
		Point* topLeft = ClosestEdgeFrom(s, (Point) {0, 0});
		Point* topRight = ClosestEdgeFrom(s, (Point) {w, 0});
		Point* bottomLeft = ClosestEdgeFrom(s, (Point) {0, h});
		Point* bottomRight = ClosestEdgeFrom(s, (Point) {w, h});
		res->topLeft = *topLeft;
		res->topRight = *topRight;
		res->bottomLeft = *bottomLeft;
		res->bottomRight = *bottomRight;
		delete topLeft;
		delete topRight;
		delete bottomLeft;
		delete bottomRight;
		return res;
	}

// Function to extract the Sudoku region from the straightened image
	Matrix*
	ExtractSudokuFromStraightImg(const Matrix& straightImage, const Square& sudokuEdges, const float rotationAngle)
	{
		const Point squareCenter = (Point) {
				(sudokuEdges.topLeft.x + sudokuEdges.topRight.x + sudokuEdges.bottomLeft.x +
				 sudokuEdges.bottomRight.x) / 4,
				(sudokuEdges.topLeft.y + sudokuEdges.topRight.y + sudokuEdges.bottomLeft.y +
				 sudokuEdges.bottomRight.y) / 4};
		Square* rotatedSquare1 = RotateSquare(sudokuEdges, squareCenter, rotationAngle);
		Square* rotatedSquare = Order(*rotatedSquare1, straightImage.cols, straightImage.rows);
		delete rotatedSquare1;

		printf("stopLeft: %i %i\n", rotatedSquare->topLeft.x, rotatedSquare->topLeft.y);
		printf("stopRight: %i %i\n", rotatedSquare->topRight.x, rotatedSquare->topRight.y);
		printf("sbottomLeft: %i %i\n", rotatedSquare->bottomLeft.x, rotatedSquare->bottomLeft.y);
		printf("sbottomRight: %i %i\n", rotatedSquare->bottomRight.x, rotatedSquare->bottomRight.y);

		const int sudoku_width = abs(rotatedSquare->topRight.x - rotatedSquare->topLeft.x);
		const int sudoku_height = abs(rotatedSquare->bottomLeft.y - rotatedSquare->topLeft.y);
		Matrix* result = new Matrix(sudoku_height, sudoku_width, 1);

		for (int y = 0; y < sudoku_height; y++)
		{
			for (int x = 0; x < sudoku_width; x++)
			{
				// Map coordinates from the original Sudoku edges to the rotated image
				Point rotatedPoint = {rotatedSquare->topLeft.x + x, rotatedSquare->topLeft.y + y};

				// Get pixel value from the straightened image and set it in the result matrix
				const float pixelValue =
						rotatedPoint.x < 0 || rotatedPoint.y < 0 || rotatedPoint.x >= straightImage.cols ||
						rotatedPoint.y >= straightImage.rows ? 0
															 : straightImage.data[rotatedPoint.y * straightImage.cols +
																				  rotatedPoint.x];
				result->data[y * result->cols + x] = pixelValue;
			}
		}

		delete rotatedSquare;

		return result;
	}

	Matrix** Split(const Matrix& matrix)
	{
		Matrix** res = new Matrix* [81];
		const int cellW = matrix.cols / 9;
		const int cellH = matrix.rows / 9;

		for (int j = 0; j < 9; j++)
		{
			const int jY = j * cellH;
			for (int i = 0; i < 9; i++)
			{
				const int iX = i * cellW;
				Matrix* cell = new Matrix(cellH, cellW, 1);
				for (int k = 0; k < cellW; k++)
					for (int l = 0; l < cellH; l++)
						cell->data[l * cell->cols + k] = matrix.data[(jY + l) * matrix.cols + iX + k];
				res[j * 9 + i] = cell;
			}
		}

		return res;
	}

}
