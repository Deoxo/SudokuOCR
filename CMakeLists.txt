cmake_minimum_required(VERSION 3.5)

project(SudokuOCR VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/home/mat/Qt/6.6.1/gcc_64/lib/cmake/")
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets LinguistTools)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets LinguistTools)

set(TS_FILES SudokuOCR_fr_FR.ts)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

#add_compile_definitions(USE_SSE2)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -msse3")

#add_compile_definitions(USE_SSE3)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")

add_compile_definitions(USE_AVX2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG_MODE)
else ()
    add_compile_definitions(RELEASE_MODE)
endif ()

set(PROJECT_SOURCES
        main.cpp
        GUI/mainwindow.cpp
        GUI/mainwindow.h
        GUI/mainwindow.ui
        Tools/Matrix.h
        Tools/Matrix.cpp
        Imagery/Imagery.cpp
        Imagery/Imagery.h
        Imagery/GridDetection.cpp
        Imagery/GridDetection.h
        Tools/Settings.h
        Tools/List.cpp
        Tools/List.h
        NeuralNetwork/Network.cpp
        NeuralNetwork/Network.h
        NeuralNetwork/Layer.cpp
        NeuralNetwork/Layer.h
        NeuralNetwork/NetworkUtils.cpp
        NeuralNetwork/NetworkUtils.h
        NeuralNetwork/DatasetManager.cpp
        NeuralNetwork/DatasetManager.h
        ${TS_FILES}
        Tools/FileManagement.cpp
        Tools/FileManagement.h
        NeuralNetwork/ActivationFunction.cpp
        NeuralNetwork/ActivationFunction.h
        NeuralNetwork/CostFunction.cpp
        NeuralNetwork/CostFunction.h
        NeuralNetwork/Optimizer.cpp
        NeuralNetwork/Optimizer.h
        Tools/Solver.cpp
        Tools/Solver.h
)
include_directories(${PROJECT_SOURCE_DIR})
if (${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(SudokuOCR
            MANUAL_FINALIZATION
            ${PROJECT_SOURCES}
            GUI/ImgSelectorButton.h GUI/ImgSelectorButton.cpp
            GUI/ImageDisplay.h GUI/ImageDisplay.cpp
            Core.h Core.cpp
            GUI/ShapeDefiner.h GUI/ShapeDefiner.cpp


    )
    # Define target properties for Android with Qt 6 as:
    #    set_property(TARGET SudokuOCR APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
    #                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
    # For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation

    qt_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
else ()
    if (ANDROID)
        add_library(SudokuOCR SHARED
                ${PROJECT_SOURCES}
        )
        # Define properties for Android with Qt 5 after find_package() calls as:
        #    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else ()
        add_executable(SudokuOCR
                ${PROJECT_SOURCES}
        )
    endif ()

    qt5_create_translation(QM_FILES ${CMAKE_SOURCE_DIR} ${TS_FILES})
endif ()

target_link_libraries(SudokuOCR PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if (${QT_VERSION} VERSION_LESS 6.1.0)
    set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.SudokuOCR)
endif ()
set_target_properties(SudokuOCR PROPERTIES
        ${BUNDLE_ID_OPTION}
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
        MACOSX_BUNDLE TRUE
        WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS SudokuOCR
        BUNDLE DESTINATION .
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if (QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(SudokuOCR)
endif ()
