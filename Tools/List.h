//
// Created by mat on 03/10/23.
//

#ifndef S3PROJECT_LIST_H
#define S3PROJECT_LIST_H
typedef struct List
{
    void* data;
    struct List* next;
} List;

List* ListCreate();

void ListAdd(List* list, void* data);

void* ListGet(const List* list, int index);

void ListSet(List* list, int index, void* data);

void ListRemove(List* list, int index);

List* ListGetList(const List* list, int index);

void ListDeepFree(List* list);

void ListFree(List* list);

#endif //S3PROJECT_LIST_H
