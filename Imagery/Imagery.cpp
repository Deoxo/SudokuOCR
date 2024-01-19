//
// Created by mat on 27/12/23.
//

#include "Imagery.h"
#include <QImage>
#include <QDir>
#include "../Tools/Settings.h"
#include "Tools/FileManagement.h"
#include "Tools/Math.h"
#include <iostream>
#include <stack>

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
				matrix->data[i * matrix->cols + j + 0 * matrix->matrixSize] = (float) (qRed(pixel));   // Red channel
				matrix->data[i * matrix->cols + j + 1 * matrix->matrixSize] = (float) (qGreen(pixel)); // Green channel
				matrix->data[i * matrix->cols + j + 2 * matrix->matrixSize] = (float) (qBlue(pixel));  // Blue channel
			}
		}

		return matrix;
	}

	Matrix* ConvertToGrayscale(const Matrix& m)
	{
		Matrix* gray = new Matrix(m.rows, m.cols);
		for (int i = 0; i < m.matrixSize; i++)
			gray->data[i] = (float)
					(0.299 * m.data[i] + 0.587 * m.data[m.matrixSize + i] + 0.114 * m.data[2 * m.matrixSize + i]);
		return gray;
	}

	Matrix* GetGaussianKernel(const int size)
	{
		Matrix* kernel = new Matrix(size, size);

		double sum = 0;
		int i, j;

		for (i = 0; i < size; i++)
		{
			for (j = 0; j < size; j++)
			{
				double x = i - (size - 1) / 2.0;
				double y = j - (size - 1) / 2.0;
				kernel->data[i * size + j] =
						GAUSS_BLUR_K * exp(((pow(x, 2) + pow(y, 2)) / ((2 * pow(GAUSS_BLUR_SIGMA, 2)))) * (-1));
				sum += kernel->data[i * size + j];
			}
		}

		*kernel /= (float) sum;

		return kernel;
	}

	Matrix* Blur(const Matrix& m, const int strength)
	{
		Matrix* kernel = GetGaussianKernel(strength);
		Matrix* res = Matrix::ValidConvolution(m, *kernel);

		delete kernel;

		return res;
	}

	void
	AdaptiveThreshold(const Matrix& m, Matrix& output, const int neighborhoodSize, const float offset,
					  const float highValue, const float lowValue)
	{
		// Adaptive threshold with gaussian weighted sum of the neighbourhood values
		Matrix* kernel = GetGaussianKernel(neighborhoodSize);
		Matrix* thresh = Matrix::ValidConvolution(m, *kernel);
		for (int i = 0; i < output.matrixSize; i++)
			output.data[i] = m.data[i] > thresh->data[i] - offset ? highValue : lowValue;

		delete thresh;
		delete kernel;
	}

	void Dilate(const Matrix& m, Matrix& output, const int neighborhoodSize)
	{
		Matrix kernel(neighborhoodSize, neighborhoodSize);
		for (int i = 0; i < kernel.matrixSize; i++)
			kernel.data[i] = 0;
		for (int i = 0; i < neighborhoodSize; i++)
		{
			kernel.data[i * neighborhoodSize + neighborhoodSize / 2] = 1;
			kernel.data[(neighborhoodSize / 2) * neighborhoodSize + i] = 1;
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

		const int filterCols = kernel.cols;
		const int filterRows = kernel.rows;

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
							const float v = m.data[inputRow * m.cols + inputCol] * kernel.data[k * kernel.cols + l];
							if (v > maxi)
								maxi = v;
						}
					}
				}
				output.data[i * output.cols + j] = maxi;
			}
		}
	}

	void Erode(const Matrix& m, Matrix& output, const int neighborhoodSize)
	{
		Matrix kernel(neighborhoodSize, neighborhoodSize);
		for (int i = 0; i < kernel.matrixSize; i++)
			kernel.data[i] = 0;
		for (int i = 0; i < neighborhoodSize; i++)
		{
			kernel.data[i * neighborhoodSize + neighborhoodSize / 2] = 1;
			kernel.data[(neighborhoodSize / 2) * neighborhoodSize + i] = 1;
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

		const int filterCols = kernel.cols;
		const int filterRows = kernel.rows;

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
							const float ke = kernel.data[k * kernel.cols + l];
							const float v = m.data[inputRow * m.cols + inputCol] * ke;
							if (ke != 0 && v < mini)
								mini = v;
						}
					}
				}
				output.data[i * output.cols + j] = mini;
			}
		}
	}

	void BitwiseNot(const Matrix& m, Matrix& output)
	{
		for (int i = 0; i < m.matrixSize; i++)
			output.data[i] = 255 - m.data[i];
	}

	void Canny(const Matrix& m, Matrix& output, const float lowThreshold, const float highThreshold)
	{
		// Canny edge detection algorithm
		Matrix* blurred = Blur(m, GAUSS_BLUR_SIZE);
		Matrix** gradients = GetGradients(*blurred);
		*gradients[0] *= 255.f / gradients[0]->Max();
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
		Matrix horizontalKernel(3, 3, {-1, 0, 1, -2, 0, 2, -1, 0, 1});
		Matrix verticalKernel(3, 3, {-1, -2, -1, 0, 0, 0, 1, 2, 1});

		// Convolve the kernels with the matrix
		Matrix* ix = Matrix::ValidConvolution(m, horizontalKernel);;
		Matrix* iy = Matrix::ValidConvolution(m, verticalKernel);

		// Create the gradients array
		Matrix** gradients = new Matrix* [2];
		gradients[0] = Matrix::CreateSameSize(m);
		gradients[1] = Matrix::CreateSameSize(m);

		// G
		for (int i = 0; i < ix->matrixSize; i++)
		{
			const float ix2 = ix->data[i] * ix->data[i];
			const float iy2 = iy->data[i] * iy->data[i];
			gradients[0]->data[i] = std::sqrt(ix2 + iy2);
		}

		// Theta
		for (int i = 0; i < ix->matrixSize; i++)
			gradients[1]->data[i] = std::atan2(iy->data[i], ix->data[i]);

		// Free the kernels
		delete ix;
		delete iy;

		return gradients;
	}

	void NonMaximumSuppression(const Matrix& m, const Matrix& angles, Matrix& output)
	{
		m.CopyValuesTo(output);
		return;
		// Convert angles to degrees in [0, 180[
		for (int i = 0; i < angles.matrixSize; i++)
		{
			angles.data[i] = (float) (angles.data[i] * 180 / M_PIf);
			if (angles.data[i] < 0)
				angles.data[i] += 180;
			if (angles.data[i] < 0 || angles.data[i] > 180)
				printf("angle: %f\n", angles.data[i]);
		}

		// Non maximum suppression
		for (int y = 1; y < m.rows - 1; y++)
		{
			for (int x = 1; x < m.cols - 1; x++)
			{
				const float angle = angles.data[y * angles.cols + x];
				const float v = m.data[y * m.cols + x];
				float v1, v2;

				if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180))
				{
					v1 = m.data[y * m.cols + x - 1];
					v2 = m.data[y * m.cols + x + 1];
				}
				else if (angle >= 22.5 && angle < 67.5)
				{
					v1 = m.data[(y - 1) * m.cols + x + 1];
					v2 = m.data[(y + 1) * m.cols + x - 1];
				}
				else if (angle >= 67.5 && angle < 112.5)
				{
					v1 = m.data[(y - 1) * m.cols + x];
					v2 = m.data[(y + 1) * m.cols + x];
				}
				else
				{
					v1 = m.data[(y - 1) * m.cols + x - 1];
					v2 = m.data[(y + 1) * m.cols + x + 1];
				}

				output.data[y * output.cols + x] = v < v1 || v < v2 ? 0 : v;
			}
		}
	}


	void DoubleThreshold(const Matrix& m, Matrix& output, const float lowThreshold, const float highThreshold)
	{
		for (int i = 0; i < m.matrixSize; i++)
		{
			const float v = m.data[i];
			output.data[i] = v < lowThreshold ? 0.f : (v > highThreshold ? 255.f : 127.f);
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
		Matrix* accumulator = new Matrix(accumulatorRows, accumulatorCols);

		// Precompute cos and sin values
		float* cosTable = new float[180];
		float* sinTable = new float[180];
		for (int i = 0; i < 180; i++)
		{
			cosTable[i] = std::cos((float) i * M_PIf / 180);
			sinTable[i] = std::sin((float) i * M_PIf / 180);
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
						const int rho =
								(int) std::round((float) y * cosTable[theta] + (float) x * sinTable[theta]) + diagonal;
						accumulator->data[rho * accumulator->cols + theta] += 1;
					}
				}
			}
		}

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

		while (changed && iters < maxIter)
		{
			iters++;
			changed = 0;
			int cluster1Length = 0;
			int cluster2Length = 0;
			for (int i = 0; i < dataLength; i++)
			{
				const int v = data[i];
				const float d1 = std::abs((float) v - clusterCenters[0]);
				const float d2 = std::abs((float) v - clusterCenters[1]);
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
				clusterCenters[1] = (float) sum2 / (float) cluster2Length;
				clusterCenters[0] = clusterCenters[1];
				changed = 1;
			}
			else if (cluster2Length == 0)
			{
				clusterCenters[0] = (float) sum1 / (float) cluster1Length;
				clusterCenters[1] = clusterCenters[0];
				changed = 1;
			}
			else
			{
				const float newCenter1 = (float) sum1 / (float) cluster1Length;
				const float newCenter2 = (float) sum2 / (float) cluster2Length;
				if (newCenter1 != clusterCenters[0] || newCenter2 != clusterCenters[1])
				{
					changed = 1;
					clusterCenters[0] = newCenter1;
					clusterCenters[1] = newCenter2;
				}
			}
		}

		for (int i = 0; i < dataLength; i++)
		{
			const int v = data[i];
			const float d1 = std::abs((float) v - clusterCenters[0]);
			const float d2 = std::abs((float) v - clusterCenters[1]);
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
			angles[i] = (int) (lines[i].theta * 180 / M_PIf); // convert to degrees

			// Remap angle to [-45, 45]
			if (angles[i] > 90)
			{
				if (angles[i] > 135)
					angles[i] = -(90 - angles[i] % 90);
				else
					angles[i] = angles[i] % 90;
			}
			else if (angles[i] > 45)
				angles[i] = -(90 - angles[i]);
			avg += (float) angles[i];
		}

		avg /= (float) numLines;
		for (int i = 0; i < numLines; ++i)
			if (std::abs((float) angles[i] - avg) < 15)
			{
				avg2 += (float) angles[i];
				avg2n++;
			}
		avg2 /= avg2n;

		delete[] angles;
		return avg2 > 45 ? -(90 - avg2) : avg2;
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

							const float spatial_distance = sqrtf(powf((float) i, 2.f) + powf((float) j, 2.f));
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
		const Matrix sobel_x(3, 3, {-1, 0, 1, -2, 0, 2, -1, 0, 1});
		const Matrix sobel_y(3, 3, {-1, -2, -1, 0, 0, 0, 1, 2, 1});

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
						gx += m.data[(y + j) * width + (x + i)] * sobel_x.data[(j + 1) * 3 + i + 1];
						gy += m.data[(y + j) * width + (x + i)] * sobel_y.data[(j + 1) * 3 + i + 1];
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
		int count;
		int x1, y1, x2, y2;

		// Compute integral image
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

		// Compute threshold
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
				if (m.data[j * m.cols + i] * (float) count < sum * (1.0 - threshold))
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
		const float dist =
				(float) (p1->x - p2->x) * (float) (p1->x - p2->x) + (float) (p1->y - p2->y) * (float) (p1->y - p2->y);
		if (dist == 0)
			return 0;
		const float vx = (float) (p2->x - p1->x) / dist;
		const float vy = (float) (p2->y - p1->y) / dist;

		int numBlackPixels = 0;
		for (int i = 1; i < SEGMENT_BLACK_PERCENTAGE_NUM_SAMPLES; ++i)
		{
			const int px = (int) ((float) p1->x + vx * (dist / (float) i));
			const int py = (int) ((float) p1->y + vy * (dist / (float) i));
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

		printf("%i %i\n", (int) ((float) point->x * xRatio), (int) ((float) point->y * yRatio));
	}

	float StandardDeviation(const Matrix& m, const int blockSize)
	{
		const int width = m.cols;
		const int height = m.rows;
		const int radius = blockSize / 2;

		float sum;
		float sum2;
		int count;

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
				const float mean = sum / (float) count;
				const float variance = sum2 / (float) count - mean * mean;
				if (variance <= 0)
					continue;
				const float stdDev = sqrtf(variance);
				rr += stdDev;
			}
			rr /= (float) width - 2.f * (float) radius;
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

	// Compute the dispersion of the image (ie if there is a lot of small (less than 128) differences between pixels)
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
		const int cropSize = (int) ((float) cellSize * (float) cropPercentage / 100.f);
		const int croppedCellSize = cellSize - 2 * cropSize;
		Matrix** croppedCells = new Matrix* [numCells];

		for (int i = 0; i < numCells; ++i)
		{
			croppedCells[i] = new Matrix(croppedCellSize, croppedCellSize);
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
		const int d = (int) (28.f * BORDER_CROP_PERCENTAGE / 100.f);
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

	// Determine if cells are empty by compute the occupancy of the center of the cell
	bool* GetEmptyCells(const Matrix** cells, const float emptinessThreshold)
	{
		bool* emptyCells = new bool[81];
		for (int i = 0; i < 81; ++i)
		{
			const int r = cells[i]->rows / 1.4f;
			const int c = cells[i]->cols / 1.4f;
			const int dr = (cells[i]->rows - r) / 2;
			const int dc = (cells[i]->cols - c) / 2;
			Matrix center(r, c);
			for (int j = 0; j < r; ++j)
				for (int k = 0; k < c; ++k)
					center.data[j * center.cols + k] = cells[i]->data[(j + dr) * cells[i]->cols + k + dc];

			emptyCells[i] = center.Sum() < emptinessThreshold * (float) cells[i]->matrixSize * 255;
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
		{
			resizedCells[i] = new Matrix(resizedCellSize, resizedCellSize);
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


	// Delete a pixel and recursively delete its neighbors within a given window if they are white.
	void
	RemoveContinuousPixels(Matrix& img, const int oy, const int ox, const int halfWindowSize)
	{
		// Use stack instead of recursion to avoid stack overflow
		std::stack<Point> stack;
		stack.emplace(ox, oy);

		while (!stack.empty())
		{
			const Point p = stack.top();
			stack.pop();
			const int x = p.x;
			const int y = p.y;
			if (y < 0 || y >= img.rows || x < 0 || x >= img.cols)
				continue;
			if (img.data[y * img.cols + x] == 0)
				continue;

			img.data[y * img.cols + x] = 0;

			for (int y_ = -halfWindowSize; y_ <= halfWindowSize; ++y_)
				for (int x_ = -halfWindowSize; x_ <= halfWindowSize; ++x_)
					stack.emplace(x + x_, y + y_);
		}
	}

	// Remove lines of a Sudoku by recursively removing neighbor pixels along the borders of the image
	void RemoveLines(Matrix& img)
	{
		const int halfWindowSize = std::min(1, std::min(img.cols, img.rows) / 300);
		for (int x = 0; x < img.cols; ++x)
		{
			RemoveContinuousPixels(img, 0, x, halfWindowSize);
			RemoveContinuousPixels(img, img.rows - 1, x, halfWindowSize);
		}
		for (int y = 0; y < img.rows; ++y)
		{
			RemoveContinuousPixels(img, y, 0, halfWindowSize);
			RemoveContinuousPixels(img, y, img.cols - 1, halfWindowSize);
		}
	}

	Matrix* Rotation(const Matrix& matrix, const Square& s, const double degree)
	{
		double angle = (-1) * degree * (M_PIf / 180);
		double cos_val = cos(angle);
		double sin_val = sin(angle);

		double center_x = (s.topLeft.x + s.topRight.x + s.bottomLeft.x + s.bottomRight.x) / 4.;
		double center_y = (s.topLeft.y + s.topRight.y + s.bottomLeft.y + s.bottomRight.y) / 4.;
		//double center_x = matrix->cols / 2;
		//double center_y = matrix->rows / 2;

		Matrix* res = new Matrix(matrix.rows, matrix.cols);
		for (int i = 0; i < matrix.rows; i++)
			for (int j = 0; j < matrix.cols; j++)
				res->data[i * res->cols + j] = 0;

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

		return res;
	}

	// Compute the rotated coordinates of a point around another point
	void RotatePoint(const Point& pt, const Point& center, const float angle, Point& res)
	{
		const float radians = angle * M_PIf / 180.0f;
		const float ptxf = (float) (pt.x);
		const float ptyf = (float) (pt.y);
		const float ctxf = (float) (center.x);
		const float ctyf = (float) (center.y);
		const int rotated_x = (int) ((ptxf - ctxf) * std::cos(radians) - (ptyf - ctyf) * std::sin(radians) + ctxf);
		const int rotated_y = (int) ((ptxf - ctxf) * std::sin(radians) + (ptyf - ctyf) * std::cos(radians) + ctyf);

		res.x = rotated_x;
		res.y = rotated_y;
	}

	// Rotate the vertices of a square by a certain angle around a given point
	Square* RotateSquare(const Square& s, const Point& center, const float angle)
	{
		Square* res = new Square();
		RotatePoint(s.topLeft, center, angle, res->topLeft);
		RotatePoint(s.topRight, center, angle, res->topRight);
		RotatePoint(s.bottomLeft, center, angle, res->bottomLeft);
		RotatePoint(s.bottomRight, center, angle, res->bottomRight);
		return res;
	}

	// Compute the vertex of a square that is the closest from a given point
	Point* ClosestEdgeFrom(const Square& s, const Point& pt)
	{
		Point* res = new Point(pt);

		float minDist = Dist(s.topLeft, pt);
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

	// Approximately order the vertices of a square
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

	// Split the Sudoku in 81 cells
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
				res[j * 9 + i] = new Matrix(cellH, cellW);
				Matrix* cell = res[j * 9 + i];
				for (int k = 0; k < cellW; k++)
					for (int l = 0; l < cellH; l++)
						cell->data[l * cell->cols + k] = matrix.data[(jY + l) * matrix.cols + iX + k];
			}
		}

		return res;
	}

	// Builds a perspective matrix that maps the sudoku edges to the desired edges
	// Reference: https://github.com/opencv/opencv/blob/11b020b9f9e111bddd40bffe3b1759aa02d966f0/modules/imgproc/src/imgwarp.cpp#L3001
	Matrix* BuildPerspectiveMatrix(const Square& sudokuEdges, const Square& desiredEdges)
	{
		const float tlx = (float) sudokuEdges.topLeft.x;
		const float tly = (float) sudokuEdges.topLeft.y;
		const float trx = (float) sudokuEdges.topRight.x;
		const float try_ = (float) sudokuEdges.topRight.y;
		const float blx = (float) sudokuEdges.bottomLeft.x;
		const float bly = (float) sudokuEdges.bottomLeft.y;
		const float brx = (float) sudokuEdges.bottomRight.x;
		const float bry = (float) sudokuEdges.bottomRight.y;
		const float dTlx = (float) desiredEdges.topLeft.x;
		const float dTly = (float) desiredEdges.topLeft.y;
		const float dTrx = (float) desiredEdges.topRight.x;
		const float dTry = (float) desiredEdges.topRight.y;
		const float dBlx = (float) desiredEdges.bottomLeft.x;
		const float dBly = (float) desiredEdges.bottomLeft.y;
		const float dBrx = (float) desiredEdges.bottomRight.x;
		const float dBry = (float) desiredEdges.bottomRight.y;

		Matrix a(8, 8);
		Matrix b(8, 1);

		// First row
		a.data[0 * 8 + 0] = tlx;
		a.data[0 * 8 + 1] = tly;
		a.data[0 * 8 + 2] = 1;
		a.data[0 * 8 + 3] = a.data[0 * 8 + 4] = a.data[0 * 8 + 5] = 0;
		a.data[0 * 8 + 6] = -tlx * dTlx;
		a.data[0 * 8 + 7] = -tly * dTlx;
		b.data[0 * b.cols + 0] = dTlx;
		// Second row
		a.data[1 * 8 + 0] = trx;
		a.data[1 * 8 + 1] = try_;
		a.data[1 * 8 + 2] = 1;
		a.data[1 * 8 + 3] = a.data[1 * 8 + 4] = a.data[1 * 8 + 5] = 0;
		a.data[1 * 8 + 6] = -trx * dTrx;
		a.data[1 * 8 + 7] = -try_ * dTrx;
		b.data[1] = dTrx;
		// Third row
		a.data[2 * 8 + 0] = blx;
		a.data[2 * 8 + 1] = bly;
		a.data[2 * 8 + 2] = 1;
		a.data[2 * 8 + 3] = a.data[2 * 8 + 4] = a.data[2 * 8 + 5] = 0;
		a.data[2 * 8 + 6] = -blx * dBlx;
		a.data[2 * 8 + 7] = -bly * dBlx;
		b.data[2] = dBlx;
		// Fourth row
		a.data[3 * 8 + 0] = brx;
		a.data[3 * 8 + 1] = bry;
		a.data[3 * 8 + 2] = 1;
		a.data[3 * 8 + 3] = a.data[3 * 8 + 4] = a.data[3 * 8 + 5] = 0;
		a.data[3 * 8 + 6] = -brx * dBrx;
		a.data[3 * 8 + 7] = -bry * dBrx;
		b.data[3] = dBrx;
		// Fifth row
		a.data[4 * 8 + 0] = a.data[4 * 8 + 1] = a.data[4 * 8 + 2] = 0;
		a.data[4 * 8 + 3] = tlx;
		a.data[4 * 8 + 4] = tly;
		a.data[4 * 8 + 5] = 1;
		a.data[4 * 8 + 6] = -tlx * dTly;
		a.data[4 * 8 + 7] = -tly * dTly;
		b.data[4] = dTly;
		// Sixth row
		a.data[5 * 8 + 0] = a.data[5 * 8 + 1] = a.data[5 * 8 + 2] = 0;
		a.data[5 * 8 + 3] = trx;
		a.data[5 * 8 + 4] = try_;
		a.data[5 * 8 + 5] = 1;
		a.data[5 * 8 + 6] = -trx * dTry;
		a.data[5 * 8 + 7] = -try_ * dTry;
		b.data[5] = dTry;
		// Seventh row
		a.data[6 * 8 + 0] = a.data[6 * 8 + 1] = a.data[6 * 8 + 2] = 0;
		a.data[6 * 8 + 3] = blx;
		a.data[6 * 8 + 4] = bly;
		a.data[6 * 8 + 5] = 1;
		a.data[6 * 8 + 6] = -blx * dBly;
		a.data[6 * 8 + 7] = -bly * dBly;
		b.data[6] = dBly;
		// Eighth row
		a.data[7 * 8 + 0] = a.data[7 * 8 + 1] = a.data[7 * 8 + 2] = 0;
		a.data[7 * 8 + 3] = brx;
		a.data[7 * 8 + 4] = bry;
		a.data[7 * 8 + 5] = 1;
		a.data[7 * 8 + 6] = -brx * dBry;
		a.data[7 * 8 + 7] = -bry * dBry;
		b.data[7] = dBry;

		// Solve the system of linear equations
		Matrix* solution = Math::GaussianElimination(a, b, 8);
		Matrix* res = new Matrix(3, 3);
		std::copy(solution->data, solution->data + 8, res->data);
		res->data[8] = 1;

		return res;
	}

	// Apply the inverse perspective transformation to a point (ie. map the point from the output image to the input image)
	Point ApplyInversePerspectiveTransformation(const Matrix& inversePerspectiveMatrix, const Point& p)
	{
		// Create the point in homogeneous coordinates
		Matrix homogeneousPoint(3, 1);
		homogeneousPoint.data[0] = (float) p.x;
		homogeneousPoint.data[1] = (float) p.y;
		homogeneousPoint.data[2] = 1.f;

		// Apply the inverse perspective transformation
		Matrix out(3, 1);
		Matrix::Multiply(inversePerspectiveMatrix, homogeneousPoint, out);

		// Divide by the last coordinate to get the real coordinates
		Point res;
		res.x = (int) (out.data[0] / out.data[2]);
		res.y = (int) (out.data[1] / out.data[2]);

		// Free memory

		return res;
	}

	// Apply the perspective transformation to an image
	Matrix*
	PerspectiveTransform(const Matrix& img, const Square& sudokuEdges, const Square& desiredEdges, const int squareSize)
	{
		// Compute perspective matrices
		Matrix* perspectiveMatrix = BuildPerspectiveMatrix(sudokuEdges, desiredEdges);
		Matrix* inversePerspectiveMatrix = Math::Get3x3MatrixInverse(*perspectiveMatrix);

		// Apply the perspective transformation
		Matrix* res = new Matrix(squareSize, squareSize);
		for (int y = 0; y < res->rows; y++)
		{
			for (int x = 0; x < res->cols; x++)
			{
				Point p = {x, y};
				Point originalPoint = ApplyInversePerspectiveTransformation(*inversePerspectiveMatrix, p);
				if (originalPoint.x < 0 || originalPoint.y < 0 || originalPoint.x >= img.cols ||
					originalPoint.y >= img.rows)
					res->data[y * res->cols + x] = 0;
				else
					res->data[y * res->cols + x] = img.data[originalPoint.y * img.cols + originalPoint.x];
			}
		}

		// Free memory
		delete perspectiveMatrix;
		delete inversePerspectiveMatrix;

		return res;
	}

	void
	FloodFill(Matrix& img, QPoint pos, const int halfWindowSize, std::list<QPoint>& group, // NOLINT(*-no-recursion)
			  const float target)
	{
		if (pos.x() < 0 || pos.x() >= img.cols || pos.y() < 0 || pos.y() >= img.rows)
			return;
		if (img.data[pos.y() * img.cols + pos.x()] != target)
			return;
		img.data[pos.y() * img.cols + pos.x()] = 0;
		group.push_back(pos);

		for (int y = -halfWindowSize; y <= halfWindowSize; ++y)
		{
			for (int x = -halfWindowSize; x <= halfWindowSize; ++x)
				FloodFill(img, QPoint(pos.x() + x, pos.y() + y), halfWindowSize, group, target);
		}
	}

	[[nodiscard]] float Dist(const QPoint& a, const QPoint& b)
	{
		const float dx = (float) (a.x() - b.x());
		const float dy = (float) (a.y() - b.y());

		return std::sqrt(dx * dx + dy * dy);
	}

	[[nodiscard]] float Dist(const Point& a, const Point& b)
	{
		const float dx = (float) (a.x - b.x);
		const float dy = (float) (a.y - b.y);

		return std::sqrt(dx * dx + dy * dy);
	}

	Matrix** CenterAndResizeDigits(const Matrix** cells, const bool* emptyCells)
	{
		Matrix** centeredDigits = new Matrix* [81];
		for (int i = 0; i < 81; ++i)
		{
			centeredDigits[i] = new Matrix(28, 28);
			if (emptyCells[i])
				continue;
			// index of first column containing at least one white pixel
			int sx = -1;
			for (int x = 0; x < cells[i]->cols && sx == -1; ++x)
			{
				for (int y = 0; y < cells[i]->rows; ++y)
					if (cells[i]->data[y * cells[i]->cols + x] != 0)
						sx = x;
			}
			// index of last column containing at least one white pixel
			int ex = -1;
			for (int x = cells[i]->cols - 1; x >= 0 && ex == -1; x--)
			{
				for (int y = 0; y < cells[i]->rows; ++y)
					if (cells[i]->data[y * cells[i]->cols + x] != 0)
						ex = x;
			}
			// index of first row containing at least one white pixel
			int sy = -1;
			for (int y = 0; y < cells[i]->rows && sy == -1; ++y)
			{
				for (int x = 0; x < cells[i]->cols; ++x)
					if (cells[i]->data[y * cells[i]->cols + x] != 0)
						sy = y;
			}
			// index of last row containing at least one white pixel
			int ey = -1;
			for (int y = cells[i]->rows - 1; y >= 0 && ey == -1; y--)
			{
				for (int x = 0; x < cells[i]->cols; ++x)
					if (cells[i]->data[y * cells[i]->cols + x] != 0)
						ey = y;
			}

			const float rx = (float) (ex - sx) / 17.f;
			const float ry = (float) (ey - sy) / 17.f;
			const float fsy = (float) sy;
			const float fsx = (float) sx;
			for (int y = 5; y < 28 - 5; ++y)
			{
				for (int x = 5; x < 28 - 5; ++x)
				{
					centeredDigits[i]->data[y * 28 + x] = cells[i]->data[
							(int) (fsy + ((float) y - 5) * ry) * cells[i]->cols + (int) (fsx + (float) (x - 5) * rx)];
				}
			}
		}

		return centeredDigits;
	}
}