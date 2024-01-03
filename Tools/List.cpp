//
// Created by mat on 03/10/23.
//
#include "List.h"
#include <cstdlib>

List* ListCreate()
{
    List* list = (List*) malloc(sizeof(List));
    list->data = nullptr;
    list->next = nullptr;
    return list;
}

void ListAdd(List* list, void* data)
{
    List* current = list;
    while (current->next != nullptr)
        current = current->next;
    current->next = new List();
    current->next->data = data;
    current->next->next = nullptr;
}

void ListDeepFree(List* list)
{
    List* current = list;
    while (current != nullptr)
    {
        List* next = current->next;
        free(current->data);
        free(current);
        current = next;
    }
}

void ListFree(List* list)
{
    List* current = list;
    while (current != nullptr)
    {
        List* next = current->next;
        free(current);
        current = next;
    }
}

void* ListGet(const List* list, const int index)
{
    List* current = (List*) list->next; // skip the first element (it's the head)
    for (int i = 0; i < index; i++)
        current = current->next;

    return current->data;
}

List* ListGetList(const List* list, const int index)
{
    List* current = (List*) list->next; // skip the first element (it's the head)
    for (int i = 0; i < index; i++)
        current = current->next;

    return current;
}

void ListSet(List* list, const int index, void* data)
{
    List* current = (List*) list->next; // skip the first element (it's the head)
    for (int i = 0; i < index; i++)
        current = current->next;

    current->data = data;
}

void ListRemove(List* list, const int index)
{
    List* current = (List*) list->next; // skip the first element (it's the head)
    for (int i = 0; i < index - 1; i++)
        current = current->next;

    List* next = current->next;
    current->next = next->next;
    free(next->data);
    free(next);
}

