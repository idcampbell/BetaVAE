import random
from math import sqrt, floor
import math
import json

from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

with open('params.json') as json_file:  
    params = json.load(json_file)
    basepath = params["base_path"] + "/Data/"

def draw_square(im, point, num, fill):
    draw = ImageDraw.Draw(im)
    x1 = point[0] + num
    y1 = point[1] + num
    x2 = point[0] - num
    y2 = point[1] + num
    x3 = point[0] - num
    y3 = point[1] - num
    x4 = point[0] + num
    y4 = point[1] - num
    draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=fill)
    return im

def draw_circle(im, point, radius, fill):
    draw = ImageDraw.Draw(im)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill)
    return im

def find_color(code):
    if code[0] > 0.0:
        return 'red'
    elif code[1] > 0.0:
        return 'green'
    else:
        return 'blue'

def generate_data(df, basepath):
    row = 0
    for shape in ["square", "circle"]:
        for x in range(10, 54):
            for y in range(10, 54):
                color_val = random.randint(100, 255)
                colors = [(color_val, 0, 0), (0, color_val, 0), (0, 0, color_val)]
                for color in colors:
                    color_name = find_color(color)
                    for size in range(6, 15):
                        im = Image.new('RGB', (64, 64))
                        if shape == "square":
                            im = draw_square(im, [x, y], size*.5, color)
                            filename = basepath + "square_x" + str(x) + "_y" + str(y) + "_size" + str(size) + "_" + str(color_val) + "_" + color_name + ".png"
                            im.save(filename, "PNG")
                        if shape == "circle":
                            im = draw_circle(im, [x, y], size*.5, color)
                            filename = basepath + "circle_x" + str(x) + "_y" + str(y) + "_size" + str(size) + "_" + str(color_val) + "_" + color_name + ".png"
                            im.save(filename, "PNG")
                        train_info["filename"].loc[row] = filename
                        train_info["x"].loc[row] = x
                        train_info["y"].loc[row] = y
                        train_info["shape"].loc[row] = shape
                        train_info["side_length/diameter"].loc[row] = size
                        train_info["color"].loc[row] = color_name
                        train_info["color_code"].loc[row] = color_val
                        row += 1

if __name__ == "__main__":
    train_info = pd.DataFrame(np.zeros([400000, 9]), columns=["filename", "x", "y", "shape", "side_length/diameter", "color", "color_code", "reconstruction", "latent_representation"])
    generate_data(train_info, basepath)
    train_info = train_info[(train_info.T != 0).any()]
    train_info.to_csv("myData.csv")