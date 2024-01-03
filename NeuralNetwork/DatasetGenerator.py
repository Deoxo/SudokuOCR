# import os
# import random
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont, ImageFilter
#
#
# def get_text_size(text, font):
#     temp_image = Image.new("L", (1, 1))
#     temp_draw = ImageDraw.Draw(temp_image)
#     return temp_draw.textbbox((0, 0), text, font=font)[2:]
#
#
# def generate_sudoku_dataset(fonts_folder, output_folder, num_samples, image_size=(28, 28),
#                             digit_size_percentage_range=(50, 70), rotation_range=(-5, 5),
#                             offset_range_percentage=(0, 0.1)):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     else:
#         # Remove existing files in the output folder
#         existing_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
#         for file in existing_files:
#             os.remove(os.path.join(output_folder, file))
#
#     fonts = [os.path.join(fonts_folder, file) for file in os.listdir(fonts_folder) if
#              file.endswith('.ttf') or file.endswith('.TTF')]
#
#     num_samples -= num_samples % (len(fonts) * 9)
#     num_samples_per_digit = int(num_samples / len(fonts) / 9)
#
#     for font_path in fonts:
#         print(f"Generating samples for font {font_path}")
#         font = ImageFont.truetype(font_path, size=int(image_size[0] * 0.8))
#
#         for digit in range(1, 10):
#             for sample in range(num_samples_per_digit):
#                 # Measure text size
#                 text_width, text_height = get_text_size(str(digit), font)
#
#                 image = Image.new("L", image_size, color="black")
#                 draw = ImageDraw.Draw(image)
#
#                 digit_size = random.randint(*digit_size_percentage_range) * 0.01
#                 digit_font = ImageFont.truetype(font_path, size=int(image_size[0] * digit_size))
#
#                 angle = random.uniform(*rotation_range)
#                 offset_x = random.randint(int(image_size[0] * offset_range_percentage[0]),
#                                           int(image_size[0] * offset_range_percentage[1]))
#                 offset_y = random.randint(int(image_size[1] * offset_range_percentage[0]),
#                                           int(image_size[1] * offset_range_percentage[1]))
#
#                 try:
#                     rotated_digit = Image.new("L", (image_size[0], image_size[1]), color="black")
#                     digit_draw = ImageDraw.Draw(rotated_digit)
#                     digit_draw.text(((image_size[0] - text_width) / 2 + offset_x,
#                                      (image_size[1] - text_height) / 2 + offset_y),
#                                     str(digit), font=digit_font, fill="white")
#
#                     rotated_digit = rotated_digit.rotate(angle, resample=Image.BICUBIC, fillcolor="black")
#                     image.paste(rotated_digit, (0, 0), rotated_digit)
#                     binarized_image = image.point(lambda pixel: 0 if pixel == 0 else 255, '1')
#                     output_filename = f"{output_folder}/digit_{digit}_{sample}_{round(angle, 4)}.png"
#
#                     binarized_image.save(output_filename)
#                 except Exception as e:
#                     print(f"Error generating sample for digit {digit}: {e}")
#
#
# fonts_folder_path = 'datasets/Fonts'
# output_folder_path = 'datasets/custom'
# output_test_folder_path = 'datasets/customTest'
#
# numTrainSamples = 35000
# numTestSamples = 1000
# digit_size_percentage_range = (40, 95)
# rotation_range = (-2, 2)
# offset_range_percentage = (-0.2, 0.2)
#
# generate_sudoku_dataset(fonts_folder_path, output_folder_path, num_samples=numTrainSamples,
#                         digit_size_percentage_range=digit_size_percentage_range, rotation_range=rotation_range,
#                         offset_range_percentage=offset_range_percentage)
# generate_sudoku_dataset(fonts_folder_path, output_test_folder_path, num_samples=numTestSamples,
#                         digit_size_percentage_range=digit_size_percentage_range, rotation_range=rotation_range,
#                         offset_range_percentage=offset_range_percentage)
import os
import random
from PIL import Image, ImageDraw, ImageFont


def get_text_size(text, font):
    temp_image = Image.new("L", (1, 1))
    temp_draw = ImageDraw.Draw(temp_image)
    return temp_draw.textbbox((0, 0), text, font=font)[2:]


def draw_sudoku_grid(draw, size, border_width, line_width):
    # Draw outer border
    draw.rectangle([(0, 0), size], outline="black", width=border_width)

    # Draw inner lines
    cell_size = size[0] // 9
    for i in range(1, 9):
        width = line_width if i % 3 == 0 else 1
        line_position = i * cell_size
        draw.line([(line_position, 0), (line_position, size[1])], fill="black", width=width)
        draw.line([(0, line_position), (size[0], line_position)], fill="black", width=width)


def generate_sudoku_dataset(fonts_folder, output_path, output_image_size=(1000, 1000),
                            rotation_angle_range=(-2, 2), num_outputs=10,
                            border_width_range=(5, 7), line_width_range=(1, 5),
                            digit_size_range=(2, 5)):  # Adjust the default size range as needed
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        # Remove existing files in the output folder
        existing_files = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
        for file in existing_files:
            os.remove(os.path.join(output_path, file))

    fonts = [os.path.join(fonts_folder, file) for file in os.listdir(fonts_folder)
             if file.endswith('.ttf') or file.endswith('.TTF')]

    with open("datasets/custom2/sudokus.txt", "w") as f:
        for i in range(num_outputs):
            print("Generating sudoku " + str(i))
            # Create a white canvas
            image = Image.new("RGB", output_image_size, color="white")
            draw = ImageDraw.Draw(image)

            # Draw the sudoku grid
            border_width = random.randint(*border_width_range)
            line_width = random.randint(*line_width_range)
            draw_sudoku_grid(draw, output_image_size, border_width, line_width)

            # Generate a solved sudoku (dummy example)
            sudoku_grid = [[random.choice('123456789') for _ in range(9)] for _ in range(9)]

            # Render the sudoku grid as text on the image
            text_color = "black"
            cell_size = output_image_size[0] // 9

            for row_idx, row in enumerate(sudoku_grid):
                for col_idx, digit in enumerate(row):
                    selected_font = ImageFont.truetype(random.choice(fonts), size=random.randint(*digit_size_range))
                    text_width, text_height = get_text_size(str(digit), selected_font)
                    text_position = (col_idx * cell_size + (cell_size - text_width) // 2,
                                     row_idx * cell_size + (cell_size - text_height) // 2)
                    draw.text(text_position, digit, fill=text_color, font=selected_font)

            # Apply random rotation
            rotation_angle = random.uniform(*rotation_angle_range)
            rotated_image = image.rotate(rotation_angle, expand=True, resample=Image.BICUBIC, fillcolor="white")

            # Find the bounding box of the rotated Sudoku
            bbox = rotated_image.getbbox()

            # Resize the image to fit the rotated Sudoku
            rotated_image = rotated_image.crop(bbox)

            # Save the generated image
            output_filename = os.path.join(output_path, f"sudoku_{i + 1}_{rotation_angle:.2f}.png")
            f.write(
                f"{output_filename} {''.join([str(digit) for row in sudoku_grid for digit in row])}\n")
            rotated_image.save(output_filename)


# Example usage with digit size range (20 to 40)
generate_sudoku_dataset("./datasets/Fonts", "./datasets/custom2", num_outputs=400, digit_size_range=(45, 80))
