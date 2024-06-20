from PIL import Image
# 打开SVG文件
img = Image.open('umapBATCH.svg')
# 裁剪图像
cropped_image = img.crop((100, 100, 300, 300))
# 保存裁剪后的图像
cropped_image.save('cropped.svg')