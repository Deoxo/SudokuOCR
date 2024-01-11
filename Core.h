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

	[[nodiscard]] DetectionInfo* BordersDetection(const QString& imagePath, const QString& savePath) const;

	void DigitDetection(DetectionInfo* detectionInfo, const QString& savePath);

	void StepCompletedWrapper(const Matrix& img, const QString& stepName, const QString& savePath) const;

signals:

	void StepCompleted(const QString& imgPath) const;

	void OnVerticesDetected(QPoint* vertices) const;

	void OnDigitsRecognized(const Matrix* digits) const;
};

#endif // CORE_H
