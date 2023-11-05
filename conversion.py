from PIL import Image
import os
preceding_path = "Training-Data/HC/"
dire = os.listdir("Training-Data/HC")
for i in dire:
    img = Image.open(preceding_path+i)
    new_img = os.path.splitext(i)[0]
    img.save(preceding_path+new_img+".jpg")
    os.remove(preceding_path+i)

preceding_path = "Training-Data/AD/"
dire = os.listdir("Training-Data/AD")
for i in dire:
    img = Image.open(preceding_path+i)
    new_img = os.path.splitext(i)[0]
    img.save(preceding_path+new_img+".jpg")
    os.remove(preceding_path+i)
preceding_path = "Testing_Data/HC/"
dire = os.listdir("Testing_Data/HC")
for i in dire:
    img = Image.open(preceding_path+i)
    new_img = os.path.splitext(i)[0]
    img.save(preceding_path+new_img+".jpg")
    os.remove(preceding_path+i)

preceding_path = "Testing_Data/AD/"
dire = os.listdir("Testing_Data/AD")
for i in dire:
    img = Image.open(preceding_path+i)
    new_img = os.path.splitext(i)[0]
    img.save(preceding_path+new_img+".jpg")
    os.remove(preceding_path+i)