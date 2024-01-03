//
// Created by mat on 20/09/23.
//

#ifndef S3PROJECT_MATRIX_H
#define S3PROJECT_MATRIX_H

#include <QString>

#if DEBUG_MODE
#define SAFE 1
#else
#define SAFE 0
#endif

class Matrix
{
public:
    int rows;
    int cols;
    int dims;
    int size;
    int matrixSize;
    float* data;

    Matrix(int rows, int cols, int dims);

    ~Matrix();

    void Convolution(const Matrix& filter, Matrix& output) const;

    static void ValidConvolution(const Matrix& m, const Matrix& filter, Matrix& output);

    static void FullConvolution(const Matrix& m, const Matrix& filter, Matrix& output);

    void ConvolutionWithTranspose(const Matrix& filter, Matrix& output) const;

    static void TransposeFullConvolution(const Matrix& m, const Matrix& filter, Matrix& output);

    static Matrix* CreateSameSize(const Matrix& m);

    void Print() const;

    void IntPrint() const;

    void SaveAsImg(const QString& path, const QString& imgName) const;

    static void Multiply(const Matrix& a, const Matrix& b, Matrix& output);

    static void MultiplyByTranspose(const Matrix& a, const Matrix& b, Matrix& output);

    static void TransposeMultiply(const Matrix& a, const Matrix& b, Matrix& output);

    void Reset();

    static void LinearMultiply(const Matrix& in, Matrix& output, const Matrix& other);

    void SaveToFile(FILE* file) const;

    void LoadFromFile(FILE* file);

    static void Compare(const Matrix& a, const Matrix& b);

    void GoToNextMatrix();

    void ResetOffset();

    [[nodiscard]] float Sum() const;

    void Flatten();

    static void MaxPool(const Matrix& m, int filterSize, int stride, Matrix& output);

    void CopyValuesTo(const Matrix& destination) const;

    void WriteToBinaryFile(FILE* file) const;

    Matrix& operator+=(const Matrix& other);

    Matrix& operator+=(float scalar);

    Matrix& operator*=(float scalar);

private:
    int offset;
};

#endif //S3PROJECT_MATRIX_H