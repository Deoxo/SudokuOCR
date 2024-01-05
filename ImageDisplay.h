#ifndef IMAGEDISPLAY_H
#define IMAGEDISPLAY_H

#include <QLabel>

class ImageDisplay : public QLabel
{
Q_OBJECT

public:
	ImageDisplay(QWidget* parent);

	void SetImage(const QString& imgPath);

	void resizeEvent(QResizeEvent* event);

private:
	void UpdateImage();

	QPixmap originalPixmap;
};

#endif // IMAGEDISPLAY_H
