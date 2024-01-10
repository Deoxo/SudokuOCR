#include "ShapeDefiner.h"
#include <QDebug>
#include <QGraphicsItem>
#include <QPainter>

ShapeDefiner::ShapeDefiner(QWidget* parent)
        : ImageDisplay{parent}
{
}

void ShapeDefiner::paintEvent(QPaintEvent* event)
{
    QLabel::paintEvent(event);
    QPainter painter(this);
    QPen pen(Qt::red);
    painter.setPen(pen);
    painter.setBrush(QColor(255,255,255,100));

    QPolygon polygon;
    for (const auto& pt : pts)
        polygon << pt;
    painter.drawPolygon(polygon);

    pen.setColor(QColorConstants::Svg::orange);
    painter.setPen(pen);
    painter.setBrush(QColorConstants::Svg::orange);
    for (const auto& pt : pts)
        painter.drawEllipse(pt, 5, 5);
}

void ShapeDefiner::mousePressEvent(QMouseEvent* event)
{
    for (const auto& pt : pts){
        if (QLineF(event->pos(), pt).length() < 7){
            clickedPtIndex = &pt - pts;
            break;
        }
    }
}

void ShapeDefiner::mouseMoveEvent(QMouseEvent* event)
{
    if (clickedPtIndex != -1 && rect().contains(event->pos())){
        originalPts[clickedPtIndex] = pts[clickedPtIndex] = event->pos();
        update();
    }
}

void ShapeDefiner::mouseReleaseEvent(QMouseEvent* event)
{
    clickedPtIndex = -1;
}


void ShapeDefiner::SetVertices(QPoint* vertices)
{
    // Adapt vertices to scaled img
    QPixmap p = pixmap();
    const float xRatio = p.width() / (float)originalPixmap.width();
    const float yRatio = p.height() / (float)originalPixmap.height();
    const int xDiff = geometry().width() - p.width();
    const int yDiff = geometry().height() - p.height();

    for (int i = 0; i < 4; i++)
    {
        originalPts[i] = vertices[i];
        pts[i].setX(vertices[i].x() * xRatio + xDiff / 2.f);
        pts[i].setY(vertices[i].y() * yRatio + yDiff / 2.f);
    }
}

void ShapeDefiner::resizeEvent(QResizeEvent* event)
{
    ImageDisplay::resizeEvent(event);

    // Adapt vertices to scaled img
    QPixmap p = pixmap();
    const float xRatio = p.width() / (float)originalPixmap.width();
    const float yRatio = p.height() / (float)originalPixmap.height();
    const int xDiff = geometry().width() - p.width();
    const int yDiff = geometry().height() - p.height();

    for (int i = 0; i < 4; i++)
    {
        pts[i].setX(originalPts[i].x() * xRatio + xDiff / 2.f);
        pts[i].setY(originalPts[i].y() * yRatio + yDiff / 2.f);
    }
}
