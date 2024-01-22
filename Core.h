#ifndef CORE_H
#define CORE_H

#include <QObject>
#include <QString>
#include "Imagery/Imagery.h"

class Core : public QObject
{
Q_OBJECT

public:
	Core();

	~Core() override;

	[[nodiscard]] DetectionInfo* BordersDetection(const QString& imagePath, const QString& savePath) const;

	void DigitDetection(DetectionInfo* detectionInfo, const QString& savePath);

	void StepCompletedWrapper(const Matrix& img, const QString& stepName, const QString& savePath) const;

	void SaveSudokuImage(const Matrix& sudokuMatrix, const QString& imgPath, const QString& savePath,
						 const QString& perspective0Path);

signals:

	void StepCompleted(const QString& imgPath) const;

	void OnVerticesDetected(QPoint* vertices) const;

	void OnDigitsRecognized(const Matrix* digits) const;

	void OnDigitsIsolated(const QString& isolatedDigitsPath) const;

private:
	Matrix* perspectiveMatrix, * inversePerspectiveMatrix;
	Point avgDigitCellOffset;
};

#endif // CORE_H
