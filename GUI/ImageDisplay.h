#ifndef IMAGEDISPLAY_H
#define IMAGEDISPLAY_H

#include <QLabel>
#include <QPaintEvent>

class ImageDisplay : public QLabel
{
Q_OBJECT

public:
	explicit ImageDisplay(QWidget* parent);

	void SetImage(const QString& imgPath);

protected:
	void resizeEvent(QResizeEvent* event) override;

protected:
	void UpdateImage();

	QPixmap originalPixmap;
};

#endif // IMAGEDISPLAY_H
