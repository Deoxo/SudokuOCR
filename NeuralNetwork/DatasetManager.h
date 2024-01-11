//
// Created by mat on 11/11/23.
//

#ifndef S3PROJECT_DATASETMANAGER_H
#define S3PROJECT_DATASETMANAGER_H

#include "Tools/Matrix.h"

Matrix*** LoadMnist(const char* path, int dataLength, int format2D);

Matrix*** LoadCustom(const char* path, int dataLength, int format2D);

Matrix*** LoadCustom3(const char* path, int dataLength, int format2D);

/*Matrix*** LoadAdrien(int dataLength);*/

#endif //S3PROJECT_DATASETMANAGER_H
