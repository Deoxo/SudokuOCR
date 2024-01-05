#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QCloseEvent>

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);

    ~MainWindow();

signals:

    void Quit();

public slots:
    
    void OnImageSelected(const QString& filePath);

    void GoToPage0();

    void OnImageValidated();

private:
    Ui::MainWindow* ui;
    QString imgPath;
};

#endif // MAINWINDOW_H
