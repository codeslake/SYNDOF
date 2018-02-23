import numpy as np
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
import tensorlayer as tl
from pfm_handler import pfm_handler
pfm_handler = pfm_handler() 

image_path = './data/image/'
disp_map_path = './data/disparity_map/'
interval = 4
sigma = [9, 6, 3, 0]

def get_image(path, file_name):
    return scipy.misc.imread(path + file_name, mode='RGB')

def get_disparity_map(path, file_name):
	pfm_file = open(path + file_name, 'r')
	return pfm_handler.load_pfm(pfm_file)

def get_decomposed_images(image, disp_map, interval):
	interval_distance = 256 / interval
	decomposed_disp_maps = []
	decomposed_images = []

	for i in np.arange(interval):
		start = i * interval_distance
		end = (i + 1) * interval_distance

		disp_decomposed_binary = np.logical_and(disp_map >= start, disp_map < end).astype(int)

		image_decomposed = image * np.repeat(np.expand_dims(disp_decomposed_binary, 2), 3, axis=2)
		disp_decomposed = disp_map * disp_decomposed_binary

		scipy.misc.toimage(image_decomposed, cmin=0., cmax=255.).save('image_decomposed_{}.png'.format(i))
		scipy.misc.toimage(disp_decomposed, cmin=0., cmax=255.).save('disp_decomposed_{}.png'.format(i))

		decomposed_images.append(image_decomposed)
		decomposed_disp_maps.append(disp_decomposed)

	return decomposed_images, decomposed_disp_maps

def masked_gaussian_filter(d_image, sigma):
	gaussian_filter = 
	return

def get_blurred_images(d_images, d_disp_maps):
	blurred_images = []

	for i in np.arange(len(d_images)):
	    #blurred_image = gaussian_filter(d_images[i], (sigma[i], sigma[i], 0))
	    blurred_image = masked_gaussian_filter(d_images[i], sigma[i])
	    blurred_images.append(blurred_image)
	    scipy.misc.toimage(blurred_image, cmin=0., cmax=255.).save('blurred_image_{}.png'.format(i))

	return blurred_images

def get_merged_images(b_images):
	merged_image = np.sum(np.array(b_images), axis=0)
	return merged_image

def main():
	image = get_image(image_path, '0048.png')
	scipy.misc.toimage(image, cmin=0., cmax=255.).save('image.png')
	disp_map = get_disparity_map(disp_map_path, '0048.pfm')
	disp_map_norm = ((disp_map / disp_map.max()) * 255).astype(int)

	#scipy.misc.toimage(image, cmin=0., cmax=255.).save('image.png')
	#scipy.misc.toimage(disp_map, cmin=0., cmax=disp_map.max()).save('disp.png')
	scipy.misc.toimage(disp_map_norm, cmin=0., cmax=255.).save('disp_norm.png')

	decomposed_images, decomposed_disp_maps = get_decomposed_images(image, disp_map_norm, interval)
	blurred_images = get_blurred_images(decomposed_images, decomposed_disp_maps)
	merged_image = get_merged_images(blurred_images)
	scipy.misc.toimage(merged_image, cmin=0., cmax=255.).save('merged_image.png')

if __name__ == '__main__':
    main()

