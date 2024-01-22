#include "Matrix.h"
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <QImage>
#include <QDir>

#if USE_SSE2 || USE_SSE3

#include <pmmintrin.h>

#endif
#if USE_AVX2

#include <immintrin.h>

#endif

Matrix::Matrix(const int rows, const int cols, const int dims)
{
	this->matrixSize = rows * cols;
	this->size = rows * cols * dims;
	this->rows = rows;
	this->cols = cols;
	this->dims = dims;
	this->offset = 0;
#if USE_SSE2
	this->data = (float*) _mm_malloc(sizeof(float) * this->size, 16);
#elif USE_AVX2
	this->data = (float*) _mm_malloc(sizeof(float) * this->size, 32);
#else
	this->data = new float[size];
#endif
	std::fill(data, data + size, 0);
}


Matrix::Matrix(const int rows, const int cols, const std::initializer_list<float>& data)
{
#if SAFE
	if (rows * cols != data.size())
		throw std::runtime_error("Matrix : data size does not match matrix size !");
#endif

	this->matrixSize = rows * cols;
	this->rows = rows;
	this->cols = cols;
	this->dims = 1;
	this->offset = 0;
	this->size = rows * cols * dims;
#if USE_SSE2
	this->data = (float*) _mm_malloc(sizeof(float) * this->size, 16);
#elif USE_AVX2
	this->data = (float*) _mm_malloc(sizeof(float) * this->size, 32);
#else
	this->data = new float[size];
#endif

	std::copy(data.begin(), data.end(), this->data);
}

void Matrix::Convolution(const Matrix& filter, Matrix& output) const
{
	const int filterSize = filter.rows;
	const int inputCols = cols;
	const int outputCols = (inputCols - filterSize) + 1;

#if SAFE
	const int inputRows = rows;
	const int outputRows = (inputRows - filterSize) + 1;
	if (outputCols != output.cols || output.rows != outputRows)
		throw std::runtime_error("Convolution : output matrix has not the right shape !");
#endif

	const int filterRows = filter.rows;
	const int filterCols = filter.cols;

	for (int i = 0; i < output.rows; i++)
	{
		for (int j = 0; j < output.cols; j++)
		{
			float sum = 0;
			for (int k = 0; k < filterRows; k++)
			{
				for (int l = 0; l < filterCols; l++)
				{
					sum += data[(i + k) * inputCols + j + l] * filter.data[k * filterCols + l];
				}
			}
			output.data[i * outputCols + j] = sum;
		}
	}
}

Matrix* Matrix::ValidConvolution(const Matrix& m, const Matrix& filter)
{
#if SAFE
	if (filter.rows != filter.cols || filter.rows % 2 == 0)
		throw std::runtime_error("MatValidConvolution : Filter must be square and odd sized !");
#endif

	Matrix* output = new Matrix(m.rows, m.cols);

	const int outputCols = m.cols;
	const int outputRows = m.rows;

	const int filterCols = filter.cols;
	const int filterRows = filter.rows;

	const int inputCols = m.cols;
	const int inputRows = m.rows;

	const int r = (filterRows - 1) / 2;
	const int c = (filterCols - 1) / 2;
	for (int i = 0; i < outputRows; i++)
	{
		for (int j = 0; j < outputCols; j++)
		{
			float sum = 0;
			for (int k = 0; k < filterRows; k++)
			{
				const int inputRow = i + k - r;
				if (inputRow < 0 || inputRow >= inputRows)
					continue;
				for (int l = 0; l < filterCols; l++)
				{
					const int inputCol = j + l - c;
					if (inputCol >= 0 && inputCol < inputCols)
						sum += m.data[inputRow * m.cols + inputCol] * filter.data[k * filter.cols + l];
				}
			}
			output->data[i * output->cols + j] = sum;
		}
	}

	return output;
}

void Matrix::FullConvolution(const Matrix& m, const Matrix& filter, Matrix& output)
{
	const int outputCols = m.rows + filter.cols - 1;
	const int outputRows = m.rows + filter.cols - 1;

#if SAFE
	if (output.cols != outputCols || outputRows != output.rows)
	{
		fprintf(stderr, "%s%i%s%i%s", "right shape is : (", outputRows, ",", outputCols, ")\n");
		throw std::runtime_error("MatFullConvolution : Output Matrix has not the right shape ! ");
	}

#endif
	const int filterCols = filter.cols;
	const int filterRows = filter.rows;

	const int inputCols = m.cols;
	const int inputRows = m.rows;

	const int r = filterRows - 1;
	const int c = filterCols - 1;
	for (int i = 0; i < outputRows; i++)
	{
		for (int j = 0; j < outputCols; j++)
		{
			float sum = 0;
			for (int k = 0; k < filterRows; k++)
			{
				for (int l = 0; l < filterCols; l++)
				{
					const int inputRow = i + k - r;
					const int inputCol = j + l - c;
					if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
						sum += m.data[inputRow * m.cols + inputCol] * filter.data[k * filter.cols + l];

				}
			}
			output.data[i * output.cols + j] = sum;
		}
	}
}

// Does convolution of m by filter flipped 180 degrees
void Matrix::ConvolutionWithTranspose(const Matrix& filter, Matrix& output) const
{
	const int filterSize = filter.rows;
	const int inputCols = cols;
	const int outputCols = (inputCols - filterSize) + 1;

#if SAFE
	const int inputRows = rows;
	const int outputRows = (inputRows - filterSize) + 1;
	if (outputCols != output.cols || output.rows != outputRows)
		throw std::runtime_error("Convolution : output matrix has not the right shape !");
#endif

	const int filterRows = filter.rows;
	const int filterCols = filter.cols;

	for (int i = 0; i < output.rows; i++)
	{
		for (int j = 0; j < output.cols; j++)
		{
			float sum = 0;
			for (int k = 0; k < filterRows; k++)
			{
				for (int l = 0; l < filterCols; l++)
				{
					sum += data[(i + k) * inputCols + j + l] *
						   filter.data[(filterRows - k - 1) * filterCols + filterCols - l - 1];
				}
			}
			output.data[i * outputCols + j] = sum;
		}
	}
}

void Matrix::TransposeFullConvolution(const Matrix& m, const Matrix& filter, Matrix& output)
{
	const int outputCols = m.cols + filter.rows - 1;
	const int outputRows = m.rows + filter.cols - 1;

#if SAFE
	if (output.cols != outputCols || outputRows != output.rows)
	{
		fprintf(stderr,
				"Error in MatFullConvolutionTranspose: Output Matrix has incorrect shape. Expected (%d, %d), got (%d, %d)\n",
				outputRows, outputCols, output.rows, output.cols);
		throw std::runtime_error("MatFullConvolutionTranspose: Output Matrix has not the right shape!");
	}
#endif

	const int filterCols = filter.cols;
	const int filterRows = filter.rows;

	const int inputCols = m.cols;
	const int inputRows = m.rows;

	for (int i = 0; i < outputRows; i++)
	{
		for (int j = 0; j < outputCols; j++)
		{
			float sum = 0;
			for (int k = 0; k < filterRows; k++)
			{
				for (int l = 0; l < filterCols; l++)
				{
					const int inputRow = i - k;
					const int inputCol = j - l;
					if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
						sum += m.data[inputRow * m.cols + inputCol] *
							   filter.data[(filterRows - 1 - k) * filter.cols + filterCols - 1 - l];
				}
			}
			output.data[i * output.cols + j] = sum;
		}
	}
}

Matrix* Matrix::CreateSameSize(const Matrix& m)
{
	return new Matrix(m.rows, m.cols);
}

void Matrix::Print() const
{
	printf("Matrix: %ix%i\n", rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			printf("%f ", data[i * cols + j]);
		printf("\n");
	}
}

void Matrix::IntPrint() const
{
	printf("Matrix: %ix%i\n", rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			printf("%i ", (int) data[i * cols + j]);
		printf("\n");
	}
}

Matrix::~Matrix()
{
#if USE_SSE2 || USE_SSE3 || USE_AVX2
	_mm_free(data);
#else
	delete[] data;
#endif
}

void Matrix::SaveAsImg(const QString& path, const QString& imgName) const
{
	QImage image(cols, rows, QImage::Format_Grayscale8);

	for (int y = 0; y < rows; ++y)
	{
		for (int x = 0; x < cols; ++x)
		{
			const int value = static_cast<int>(data[y * cols + x]);
			image.setPixel(x, y, qRgb(value, value, value));
		}
	}

	QString fullPath = QDir::toNativeSeparators(path + "/" + imgName + ".png");

	if (!image.save(fullPath, "PNG"))
		throw std::runtime_error("Error while saving image at " + fullPath.toStdString());
}

void Matrix::Multiply(const Matrix& a, const Matrix& b, Matrix& output)
{
#if SAFE

	if (b.rows != a.cols)
		throw std::runtime_error("Matrix have not the shape to be cross producted !");

	if (output.rows != a.rows || output.cols != b.cols)
		throw std::runtime_error("Output matrix has not the right shape !");


#endif

#if USE_AVX2
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < b.cols; j++)
		{
			__m256 sum = _mm256_setzero_ps();
			int k;
			for (k = 0; k <= a.cols - 8; k += 8)
			{
#if DEBUG_MODE
				__m256 first = _mm256_loadu_ps(&a.data[i * a.cols + k]);
#else
				__m256 first = _mm256_loadu_ps(&a.data[i * a.cols + k]);
#endif
				__m256 second = _mm256_loadu_ps(&b.data[k * b.cols + j]);
				sum = _mm256_add_ps(sum, _mm256_mul_ps(first, second));
			}

			// Horizontal addition using AVX2
			sum = _mm256_hadd_ps(sum, sum);
			sum = _mm256_hadd_ps(sum, sum);

			// Extract the result
			float result[8];
			_mm256_storeu_ps(result, sum);

			// Store the result in the output matrix
			output.data[i * output.cols + j] = result[0] + result[4];

			// Handle the remaining elements if cols is not a multiple of 8
			for (; k < a.cols; ++k)
				output.data[i * output.cols + j] += a.data[i * a.cols + k] * b.data[k * b.cols + j];
		}
	}
#elif USE_SSE2 || USE_SSE3
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < b.cols; j++)
		{
			__m128 sum = _mm_setzero_ps();
			int k;
			for (k = 0; k <= a.cols - 4; k += 4)
			{
#if DEBUG_MODE
				__m128 first = _mm_loadu_ps(&a[i * a.cols + k]);
#else
				__m128 first = _mm_load_ps(&a.data[i * a.cols + k]);
#endif
				__m128 second = _mm_loadu_ps(&b.data[k * b.cols + j]);
				sum = _mm_add_ps(sum, _mm_mul_ps(first, second));
			}

#if USE_SSE3
			// Horizontal addition using SSE3
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);

			float result;
			_mm_store_ss(&result, sum);
			output[i * output.cols + j] = result;
#else
			_mm_hadd_ps(sum, sum);
			float temp[4];
			_mm_storeu_ps(temp, sum);
			output.data[i * output.cols + j] = temp[0] + temp[1] + temp[2] + temp[3];
#endif

			// Handle the remaining elements if cols is not a multiple of 4
			for (; k < a.cols; ++k)
				output.data[i * output.cols + j] += a.data[i * a.cols + k] * b.data[k * b.cols + j];

		}
	}
#else

	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < b.cols; j++)
		{
			output[i * b.cols + j] = 0;
			for (int k = 0; k < a.cols; k++)
				output[i * output.cols + j] += a[i * a.cols + k] * b[k * b.cols + j];

		}
	}
#endif
}

void Matrix::MultiplyByTranspose(const Matrix& a, const Matrix& b, Matrix& output)
{
#if SAFE
	if (a.cols != b.cols)
		throw std::runtime_error("Error: Matrix dimensions must agree.");
	if (output.rows != a.rows || output.cols != b.rows)
		throw std::runtime_error("Error: Matrix dimensions must agree.");
#endif

#if USE_AVX2
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < b.rows; j++)
		{
			__m256 sum = _mm256_setzero_ps();
			int k;
			for (k = 0; k <= a.cols - 8; k += 8)
			{
				__m256 first = _mm256_load_ps(&a.data[i * a.cols + k]);
				__m256 second = _mm256_load_ps(&b.data[j * b.cols + k]);
				sum = _mm256_add_ps(sum, _mm256_mul_ps(first, second));
			}

			// Horizontal addition using AVX2
			sum = _mm256_hadd_ps(sum, sum);
			sum = _mm256_hadd_ps(sum, sum);

			// Extract the result
			float result[8];
			_mm256_storeu_ps(result, sum);

			// Store the result in the output matrix
			output.data[i * output.cols + j] = result[0] + result[4];

			// Handle the remaining elements if cols is not a multiple of 8
			for (; k < a.cols; ++k)
				output.data[i * output.cols + j] += a.data[i * a.cols + k] * b.data[j * b.cols + k];
		}
	}
#elif USE_SSE2 || USE_SSE3
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < b.rows; j++)
		{
			__m128 sum = _mm_setzero_ps();
			int k;
			for (k = 0; k <= a.cols - 4; k += 4)
			{
				__m128 first = _mm_load_ps(&a.data[i * a.cols + k]);
				__m128 second = _mm_load_ps(&b.data[j * b.cols + k]);
				sum = _mm_add_ps(sum, _mm_mul_ps(first, second));
			}

#if USE_SSE3
			// Horizontal addition using SSE3
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);

			// Store the result in the output matrix
			_mm_store_ss(&output[i * output.cols + j], sum);
#else
			float temp[4];
			_mm_storeu_ps(temp, sum);
			output.data[i * output.cols + j] = temp[0] + temp[1] + temp[2] + temp[3];
#endif

			// Handle the remaining elements if cols is not a multiple of 4
			for (; k < a.cols; ++k)
				output.data[i * output.cols + j] += a.data[i * a.cols + k] * b.data[j * b.cols + k];

		}
	}
#else
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < b.rows; j++)
		{
			float sum = 0;
			for (int k = 0; k < a.cols; k++)
				sum += a[i * a.cols + k] * b[j * b.cols + k];

			output.data[i * output.cols + j] = sum;
		}
	}
#endif
}

void Matrix::TransposeMultiply(const Matrix& a, const Matrix& b, Matrix& output)
{
#if SAFE
	if (a.rows != b.rows)
		throw std::runtime_error("Error: Matrix dimensions must agree.");
	if (output.rows != a.cols || output.cols != b.cols)
		throw std::runtime_error("Error: Matrix dimensions must agree.");
#endif

	//SSE appears to be slower than the naive implementation
/*#if USE_SSE2
    for (int i = 0; i < a.cols; i++)
    {
        for (int j = 0; j < b.cols; j++)
        {
            __m128 sum = _mm_setzero_ps();
            int k;
            for (k = 0; k <= a.rows - 4; k += 4)
            {
                __m128 first = _mm_set_ps(a[(k + 3) * a.cols + i], a[(k + 2) * a.cols + i],
                                          a[(k + 1) * a.cols + i], a[k * a.cols + i]);
                __m128 second = _mm_set_ps(b[(k + 3) * b.cols + j], b[(k + 2) * b.cols + j],
                                           b[(k + 1) * b.cols + j], b[k * b.cols + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(first, second));
            }

            float temp[4];
            _mm_storeu_ps(temp, sum);
            output.data[i * output.cols + j] = temp[0] + temp[1] + temp[2] + temp[3];

            // Handle the remaining elements if cols is not a multiple of 4
            for (; k < a.rows; ++k)
                output.data[i * output.cols + j] += a[k * a.cols + i] * b[k * b.cols + j];

        }
    }
#elif USE_SSE3
    for (int i = 0; i < a.cols; i++)
    {
        for (int j = 0; j < b.cols; j++)
        {
            __m128 sum = _mm_setzero_ps();
            int k;
            for (k = 0; k <= a.rows - 4; k += 4)
            {
                __m128 first = _mm_set_ps(a[(k + 3) * a.cols + i], a[(k + 2) * a.cols + i],
                                          a[(k + 1) * a.cols + i], a[k * a.cols + i]);
                __m128 second = _mm_set_ps(b[(k + 3) * b.cols + j], b[(k + 2) * b.cols + j],
                                           b[(k + 1) * b.cols + j], b[k * b.cols + j]);
                sum = _mm_add_ps(sum, _mm_mul_ps(first, second));
            }

            // Horizontal addition using SSE3
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);

            // Store the result in the output matrix
            _mm_store_ss(&output.data[i * output.cols + j], sum);

            // Handle the remaining elements if rows is not a multiple of 4
            for (; k < a.rows; ++k)
                output.data[i * output.cols + j] += a[k * a.cols + i] * b[k * b.cols + j];
        }
    }
#else*/
	for (int i = 0; i < a.cols; i++)
	{
		for (int j = 0; j < b.cols; j++)
		{
			float sum = 0;
			for (int k = 0; k < a.rows; k++)
				sum += a.data[k * a.cols + i] * b.data[k * b.cols + j];

			output.data[i * output.cols + j] = sum;
		}
	}
//#endif
}

void Matrix::Reset() // NOLINT(*-make-member-function-const)
{
	memset(data, 0, matrixSize * sizeof(float));
}

void Matrix::LinearMultiply(const Matrix& in, Matrix& output, const Matrix& other)
{
#if SAFE
	if (in.rows != other.rows || in.cols != other.cols || in.rows != output.rows || in.cols != output.cols)
		throw std::runtime_error("Error: Matrix dimensions must agree.");
#endif

#if USE_AVX2
	int i;
	for (i = 0; i + 8 < in.size; i += 8)
	{
		__m256 first = _mm256_loadu_ps(in.data + i);
		__m256 second = _mm256_loadu_ps(other.data + i);

		__m256 sum = _mm256_mul_ps(first, second);

		_mm256_storeu_ps(output.data + i, sum);
	}

	for (; i < in.size; i++)
		output.data[i] = in.data[i] * other.data[i];
#elif USE_SSE2 || USE_SSE3
	float temp[4];

	int i;
	for (i = 0; i + 4 <= in.matrixSize; i += 4)
	{
		__m128 sum;
		__m128 first = _mm_load_ps(in.data + i);
		__m128 second = _mm_load_ps(other.data + i);

		sum = _mm_mul_ps(first, second);

		_mm_storeu_ps(temp, sum);

		memcpy(output.data + i, temp, 4 * sizeof(float));
	}

	for (; i < in.matrixSize; i++)
		output.data[i] = in.data[i] * other.data[i];
#else
	for (int i = 0; i < in.size; i++)
		output.data[i] = in.data[i] * other.data[i];
#endif
}

void Matrix::SaveToFile(FILE* file) const
{
	fwrite(&rows, sizeof(int), 1, file);
	fwrite(&cols, sizeof(int), 1, file);
	fwrite(&dims, sizeof(int), 1, file);
	fwrite(&size, sizeof(int), 1, file);
	fwrite(data, sizeof(float), size, file);
}

void Matrix::LoadFromFile(FILE* file)
{
	int r, c, d, s;
	if (fread(&r, sizeof(int), 1, file) != 1)
		throw std::runtime_error("Error while loading matrix rows");
	if (fread(&c, sizeof(int), 1, file) != 1)
		throw std::runtime_error("Error while loading matrix columns");
	if (fread(&d, sizeof(int), 1, file) != 1)
		throw std::runtime_error("Error while loading matrix dimensions");
	if (fread(&s, sizeof(int), 1, file) != 1)
		throw std::runtime_error("Error while loading matrix size");

	rows = r;
	cols = c;
	dims = d;
	size = s;
	matrixSize = rows * cols;

	if (fread(data, sizeof(float), size, file) != (size_t) size)
		throw std::runtime_error("Error while loading matrix data");
}

void Matrix::Compare(const Matrix& a, const Matrix& b)
{
	if (a.rows != b.rows || a.cols != b.cols)
		throw std::runtime_error("Matrices have not the same shape !");

	for (int i = 0; i < a.matrixSize; i++)
	{
		if (std::abs(a.data[i] - b.data[i]) > 0.0001)
			throw std::runtime_error("Matrices are not equal !");
	}
}

Matrix& Matrix::operator*=(const float scalar)
{
#if USE_AVX2
	__m256 scalarSSE = _mm256_set1_ps(scalar);
	int i;
	for (i = 0; i + 8 <= size; i += 8)
	{
		__m256 first = _mm256_loadu_ps(data + i);
		__m256 result = _mm256_mul_ps(first, scalarSSE);
		_mm256_storeu_ps(data + i, result);
	}

	for (; i < size; i++)
		data[i] *= scalar;
#elif USE_SSE2 || USE_SSE3
	__m128 scalarSSE = _mm_set1_ps(scalar);
	for (int i = 0; i + 4 <= matrixSize; i += 4)
	{
		__m128 first = _mm_loadu_ps(data + i);
		__m128 result = _mm_mul_ps(first, scalarSSE);
		_mm_storeu_ps(data + i, result);
	}

	for (int i = matrixSize - matrixSize % 4; i < matrixSize; i++)
		data[i] *= scalar;
#else
	for (int i = 0; i < size; i++)
		data[i] *= scalar;
#endif

	return *this;
}

Matrix& Matrix::operator/=(float scalar)
{
#if USE_AVX2
	__m256 scalarSSE = _mm256_set1_ps(scalar);
	int i;
	for (i = 0; i + 8 <= size; i += 8)
	{
		__m256 first = _mm256_loadu_ps(data + i);
		__m256 result = _mm256_div_ps(first, scalarSSE);
		_mm256_storeu_ps(data + i, result);
	}

	for (; i < size; i++)
		data[i] /= scalar;
#elif USE_SSE2 || USE_SSE3
	__m128 scalarSSE = _mm_set1_ps(scalar);
	for (int i = 0; i + 4 <= matrixSize; i += 4)
	{
		__m128 first = _mm_loadu_ps(data + i);
		__m128 result = _mm_div_ps(first, scalarSSE);
		_mm_storeu_ps(data + i, result);
	}

	for (int i = matrixSize - matrixSize % 4; i < matrixSize; i++)
		data[i] /= scalar;
#else
	for (int i = 0; i < size; i++)
		data[i] /= scalar;
#endif

	return *this;
}

void Matrix::CopyValuesTo(const Matrix& destination) const
{
#if SAFE
	if (matrixSize != destination.matrixSize)
		throw std::runtime_error("Matrices have not the same shape !");
#endif
	memcpy(destination.data, data, matrixSize * sizeof(float));
}

void Matrix::GoToNextMatrix()
{
#if SAFE
	if (offset >= size - 1)
		throw std::runtime_error("Already at the end of matrix !");
#endif
	offset += matrixSize;
	data += matrixSize;
}

void Matrix::ResetOffset()
{
	data = data - offset;
	offset = 0;
}

float Matrix::Sum() const
{
	float sum = 0;
	for (int i = 0; i < matrixSize; i++)
		sum += data[i];

	return sum;
}

void Matrix::Flatten()
{
	rows = matrixSize * dims;
	dims = 1;
	cols = 1;
	matrixSize = rows;
}

void Matrix::MaxPool(const Matrix& m, const int filterSize, const int stride, Matrix& output)
{
	const int outputCols = (m.cols - filterSize) / stride + 1;
	const int outputRows = (m.rows - filterSize) / stride + 1;

#if SAFE
	if (output.cols != outputCols || output.rows != outputRows)
		throw std::runtime_error("MatMaxPool : Output Matrix has not the right shape ! ");
#endif

	const int inputCols = m.cols;
	const int inputRows = m.rows;

	for (int i = 0; i < outputRows; i++)
	{
		for (int j = 0; j < outputCols; j++)
		{
			float max = -INFINITY;
			for (int k = 0; k < filterSize; k++)
			{
				for (int l = 0; l < filterSize; l++)
				{
					const int inputRow = i * stride + k;
					const int inputCol = j * stride + l;
					if (inputRow >= 0 && inputRow < inputRows && inputCol >= 0 && inputCol < inputCols)
					{
						const float value = m.data[inputRow * m.cols + inputCol];
						if (value > max)
							max = value;
					}
				}
			}
			output.data[i - output.cols + j] = max;
		}
	}
}

Matrix& Matrix::operator+=(const Matrix& other)
{
#if SAFE
	if (rows != other.rows || cols != other.cols)
		throw std::runtime_error("Error: Matrix dimensions must agree.");

#endif
#if USE_AVX2
	int i;
	for (i = 0; i + 8 < size; i += 8)
	{
		__m256 a = _mm256_loadu_ps(data + i);
		__m256 b = _mm256_loadu_ps(other.data + i);

		__m256 sum = _mm256_add_ps(a, b);

		_mm256_storeu_ps(data + i, sum);
	}

	for (; i < size; i++)
		data[i] += other.data[i];
#elif USE_SSE2 || USE_SSE3
	float temp[4];

	int i;
	for (i = 0; i + 4 <= matrixSize; i += 4)
	{
		__m128 sum;
		__m128 a = _mm_load_ps(data + i);
		__m128 b = _mm_load_ps(other.data + i);

		sum = _mm_add_ps(a, b);

		_mm_storeu_ps(temp, sum);

		memcpy(data + i, temp, 4 * sizeof(float));
	}

	for (; i < matrixSize; i++)
		data[i] += +other.data[i];
#else
	for (int i = 0; i < size; i++)
		data[i] += other.data[i];
#endif

	return *this;
}

Matrix& Matrix::operator+=(const float scalar)
{
	for (int i = 0; i < size; i++)
		data[i] += scalar;
	return *this;
}

void Matrix::WriteToBinaryFile(FILE* file) const
{
	// Write shape
	fwrite(&rows, sizeof(int), 1, file);
	fwrite(&cols, sizeof(int), 1, file);
	fwrite(&dims, sizeof(int), 1, file);

	// Reset offset
	const int off = offset;
	Matrix* mm = (Matrix*) this;
	mm->ResetOffset();

	// Write data
	for (int i = 0; i < matrixSize * dims; ++i)
		fwrite(data + i, sizeof(float), 1, file);

	// Restore offset
	mm->offset = off;
	mm->data += off;
}

float Matrix::Max() const
{
	float max = -INFINITY;
	for (int i = 0; i < matrixSize; i++)
	{
		if (data[i] > max)
			max = data[i];
	}
	return max;
}

float Matrix::Min() const
{
	float min = INFINITY;
	for (int i = 0; i < matrixSize; i++)
	{
		if (data[i] < min)
			min = data[i];
	}
	return min;
}

