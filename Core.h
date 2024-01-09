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

	PreprocessInfo* Preprocess(const QString& imgPath, const QString& savePath);

	DetectionInfo* Detection(const PreprocessInfo* PreprocessInfo, const QString& savePath);

	void ProcessImage(const QString& imagePath, const QString& savPath);

	void StepCompletedWrapper(const Matrix& img, const QString& stepName, const QString& savePath);

signals:

	void StepCompleted(const QString& imgPath);
};

#endif // CORE_H
