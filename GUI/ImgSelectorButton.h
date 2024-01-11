#ifndef IMGSELECTORBUTTON_H
#define IMGSELECTORBUTTON_H

#include <QPushButton>
#include <QDropEvent>
#include <QDrag>
#include <qcoreevent.h>

class ImgSelectorButton : public QPushButton
{
Q_OBJECT

public:
	ImgSelectorButton(QWidget* parent = nullptr);

protected:
	void dropEvent(QDropEvent* event) override;

	void dragEnterEvent(QDragEnterEvent* event) override;

signals:

	void fileSelected(const QString& filePath);

private slots:

	void SelectFile();
};

#endif // IMGSELECTORBUTTON_H
