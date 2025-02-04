<h2> <center> SUDOKU OCR </center> </h2>

## Table of contents
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction-🗒️">Introduction</a>
    </li>
    <li>
      <a href="#installation-🔨">Installation</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#build">Build</a></li>
        <li><a href="#run-▶️">Run</a></li>
      </ul>
    </li>
    <li><a href="#features-⚙️">Features</a></li>
  
  </ol>
</details>

<!-- INTRODUCTION -->
## Introduction 🗒️

### What is it ?

This Sudoku OCR is a C++ program which detects and solves Sudokus in images. Provided with an input image containing a Sudoku, the program pre-processes the image, locates the Sudoku, extracts and isolates it, recognizes the digits present in the grid, and eventually compute the ouzzle solution.<br>
This project was originally made in C, as a school group project during my sutdies at EPITA. I then switched the whole project to C++ before adding/improving features. I also changed the GUI lib used, from GTK3 to Qt6.<br>

#### Credits

The vast majority of the code has been written by me, though some parts are still the same as when my group submitted our project to EPITA. As a result, you may find some code of Évrard Casamayou, Flaurian Baudron or  Mathéo Chahwan.

### Creation process
The digit recognition part is taken from the  [Neurocore Deep Learning library](https://github.com/Aur3lienH/Neurocore) that I write with a friend. I already had some experience with Qt for the GUI. The image-preprocessing and grid detection have been the real challenges here. I implemented it with the help of the develpper's best friend: Google.

## Installation 🔨

### Prerequisites

Make sure you have installed the following packages, using `sudo apt install package_name`:
- **build-essential**
- **cmake**
- **qt6-base-dev**
- **qt6-base-dev-tools**
- **qt6-tools-dev**
- **qt6-l10n-tools**


### Build

This step assumes you fulfilled all the requirements aforementioned.

Run the following commands:
```sh
cd ~/Desktop
cd SudokuOCR-main
mkdir cmake-build-release
cd cmake-build-release
cmake ..
cmake --build . --target SudokuOCR
cd ..

```

### Run ▶️

```sh
./cmake-build-release/SudokuOCR
```
Then simply follow the instructions displayed. Sample Sudoku images are available in the `Images`folder.    

## Features ⚙️
- **Image pre-processing**
  - **Grayscale**
  - **Bilateral filter**
  - **Erosion**
  - **Dilatiation**
  - **Gaussian Blur**
  - **Binarization**
- **Grid detection**
  - **Canny**
  - **Hough lines**
  - **Perspective correction**
  - **Corner detection**
- **Neural Network**
  - **Fully connected layers**
  - **Dropout layers**
  - **Multihtreaded Learning**
  - **Adam optimizer**
  - **Multiple activation functions**
  - **MSE Cost**
  - **Digit dataset generator**
  - **Network save and load in binary format**
- **Sudoku solver**
- **GUI with Qt6 and Qt Designer**