
# Define ANSI escape codes for text colors
BLACK='\033[0;30m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'

# Define ANSI escape codes for bold text colors
BOLD_BLACK='\033[1;30m'
BOLD_RED='\033[1;31m'
BOLD_GREEN='\033[1;32m'
BOLD_YELLOW='\033[1;33m'
BOLD_BLUE='\033[1;34m'
BOLD_MAGENTA='\033[1;35m'
BOLD_CYAN='\033[1;36m'
BOLD_WHITE='\033[1;37m'

# Define ANSI escape codes for background colors
BG_BLACK='\033[40m'
BG_RED='\033[41m'
BG_GREEN='\033[42m'
BG_YELLOW='\033[43m'
BG_BLUE='\033[44m'
BG_MAGENTA='\033[45m'
BG_CYAN='\033[46m'
BG_WHITE='\033[47m'

# Define ANSI escape code to reset text color and background color
RESET='\033[0m'

if /snap/clion/261/bin/cmake/linux/x64/bin/cmake --build /home/mat/Documents/Code/C++/SudokuOCR/SudokuOCR/cmake-build-release --target SudokuOCR -j 14 ; then
    for i in $(seq 1 6);
    do
        echo -e "${RED}Processing image ${YELLOW}n.$i${RESET}"
        if cmake-build-release/SudokuOCR ./Images/image_0$i.jpeg test; then echo ""
        else echo -e "${RED}Error processing image ${YELLOW}n.$i${RESET}"
        fi
        
    done
fi
