#ifndef SHAPEDEFINER_H
#define SHAPEDEFINER_H

#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include "ImageDisplay.h"

class ShapeDefiner : public ImageDisplay
{
Q_OBJECT

public:
	explicit ShapeDefiner(QWidget* parent = nullptr);

	void SetVertices(QPoint* vertices);

	QPoint* GetVertices();

protected:
	void paintEvent(QPaintEvent* event) override;

	void mousePressEvent(QMouseEvent* event) override;

	void mouseMoveEvent(QMouseEvent* event) override;

	void mouseReleaseEvent(QMouseEvent* event) override;

	void resizeEvent(QResizeEvent* event) override;

private:
	QPoint originalPts[4];
	QPoint pts[4];
	int clickedPtIndex = -1;
};

#endif // SHAPEDEFINER_H
