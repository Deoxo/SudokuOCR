#include "ImageDisplay.h"

ImageDisplay::ImageDisplay(QWidget* parent) : QLabel(parent)
{
    setMinimumSize(100, 100);
    setAlignment(Qt::AlignCenter);
}


void ImageDisplay::resizeEvent(QResizeEvent* event)
{
	if (originalPixmap.isNull())
		return;
	QSize scaledSize = originalPixmap.size();
	scaledSize.scale(size(), Qt::KeepAspectRatio);
	if (scaledSize != pixmap().size())
		UpdateImage();
}

void ImageDisplay::UpdateImage()
{
	QPixmap scaled = originalPixmap.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
	setPixmap(scaled);
}

void ImageDisplay::SetImage(const QString& imgPath)
{
	originalPixmap = QPixmap(imgPath);
	setMinimumSize(originalPixmap.size() / 3);
	UpdateImage();
}
