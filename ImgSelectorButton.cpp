#include "ImgSelectorButton.h"
#include <QMimeData>
#include <QFileDialog>
#include <QImageReader>
#include <QMessageBox>

ImgSelectorButton::ImgSelectorButton(QWidget* parent) : QPushButton(parent)
{
    setAcceptDrops(true);
    
    connect(this, &QPushButton::clicked, this, &ImgSelectorButton::SelectFile);
}

void ImgSelectorButton::dropEvent(QDropEvent* event)
{
    const QMimeData* mimeData = event->mimeData();

    // check for our needed mime type, here a file or a list of files
    if (!mimeData->hasUrls())
        return;
    QStringList pathList;
    QList<QUrl> urlList = mimeData->urls();

    // extract the local paths of the files
    const int maxAuthorizedFiles = 1;
    for (int i = 0; i < urlList.size() && i < maxAuthorizedFiles; ++i)
        pathList.append(urlList.at(i).toLocalFile());

    // Check if the file has a supported image format based on its extension

    const QString filePath = pathList[0];
    const QList<QByteArray> supportedFormats = QImageReader::supportedImageFormats();
    const QString fileExtension = QFileInfo(filePath).suffix().toLower();
    if (supportedFormats.contains(fileExtension.toLatin1()))
        emit fileSelected(pathList[0]);
    else
    {
        // Create and configure the error message dialog
        QMessageBox errorMessage;
        errorMessage.setIcon(QMessageBox::Warning);
        errorMessage.setWindowTitle("Invalid file");
        errorMessage.setText("Please select an image file.");
        errorMessage.setStandardButtons(QMessageBox::Ok);

        // Show the error message dialog
        errorMessage.exec();
    }

    event->accept();
}


void ImgSelectorButton::dragEnterEvent(QDragEnterEvent* event)
{
    // Accept relevent proposed actions
    if (event->mimeData()->hasUrls())
        event->acceptProposedAction();
}

void ImgSelectorButton::SelectFile()
{
    const QList<QByteArray> supportedFormats = QImageReader::supportedImageFormats();
    QString fileDialogFilters = "Images (";
    foreach (QByteArray format, supportedFormats)
        fileDialogFilters += "*." + format.toStdString() + " ";
    fileDialogFilters[fileDialogFilters.length() - 1] = ')';

    // Opens a file selector dialog
    QString filePath = QFileDialog::getOpenFileName(this, "Select Image", "./Images", fileDialogFilters);
    if (!filePath.isEmpty())
            emit fileSelected(filePath);
}
