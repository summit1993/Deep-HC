from utilities.common_tools import calculate_rgb_mean_and_std

img_dir = 'C:\\Users\\summit\\Desktop\\data\\morph_50000\\morph_50000_image\\'
img_mean, img_std = calculate_rgb_mean_and_std(img_dir, img_size=224)
print(img_mean, img_std)