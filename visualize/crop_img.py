#%%
from pathlib import Path
from PIL import Image,  ImageOps

paths = [i for i in Path(r'C:\Users\DEEP\PycharmProjects\NCFile\plot_img\cmip6').iterdir()
         if i.is_file()
         ]
for p in paths:
    img = Image.open(p)
    area = (0, 115, 0, 586)
    re = ImageOps.crop(img, area)
    re.save(Path(r'C:\Users\DEEP\PycharmProjects\NCFile\plot_img\cmip6\cut') / p.name)
