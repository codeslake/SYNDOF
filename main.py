import numpy as np
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
import tensorlayer as tl
from pfm_handler import pfm_handler
pfm_handler = pfm_handler() 

image_path = './data/image/'
disp_map_path = './data/disparity_map/'
save_path = './output/'
num_layers = 4
sigma = [9, 6, 3, 0]
#sigma = [0, 3, 6, 9]

def expand_dims(input_1c):
	return np.repeat(np.expand_dims(input_1c, 2), 3, axis=2)

def get_image(path, file_name):
    return scipy.misc.imread(path + file_name, mode='RGB')

def get_disparity_map(path, file_name):
	pfm_file = open(path + file_name, 'r')
	return pfm_handler.load_pfm(pfm_file)

def get_decomposed_images(image, disp_map, num_layers):
	interval_distance = 256. / num_layers
	decomposed_images = []
	decomposed_disp_maps = []
	decomposed_disp_maps_binary = []

	for i in np.arange(num_layers):
		start = i * interval_distance
		end = (i + 1) * interval_distance

		decomposed_disp_binary = np.logical_and(disp_map >= start, disp_map < end).astype(int)

		decomposed_image = image * np.repeat(np.expand_dims(decomposed_disp_binary, 2), 3, axis=2)
		decomposed_disp = disp_map * decomposed_disp_binary

		decomposed_images.append(decomposed_image)
		decomposed_disp_maps.append(decomposed_disp)
		decomposed_disp_binary = decomposed_disp_binary
		decomposed_disp_maps_binary.append(decomposed_disp_binary)

		scipy.misc.toimage(decomposed_image, cmin=0., cmax=255.).save(save_path + '{}_1_image_decomposed.png'.format(i))
		scipy.misc.toimage(decomposed_disp, cmin=0., cmax=255.).save(save_path + '{}_2_disp_decomposed.png'.format(i))
		scipy.misc.toimage(decomposed_disp_binary, cmin=0., cmax=1.).save(save_path + '{}_3_disp_decomposed_binary.png'.format(i))
		scipy.misc.toimage(1-decomposed_disp_binary, cmin=0., cmax=1.).save(save_path + '{}_4_disp_decomposed_binary_flip.png'.format(i))

	return decomposed_images, decomposed_disp_maps_binary

def masked_gaussian_filter(d_image, mask, sigma, i):
	blurred_image = gaussian_filter((d_image / 255.).astype(float), (sigma, sigma, 0))

	blurred_mask = gaussian_filter(mask.astype(float), (sigma, sigma))
	blurred_mask_3c = np.repeat(np.expand_dims(blurred_mask, 2), 3, axis = 2)
	blurred_mask_3c_epsilon = np.copy(blurred_mask_3c)
	blurred_mask_3c_epsilon[np.where(blurred_mask_3c_epsilon == 0)] = 0.00001

	masked_conv_image = blurred_image / blurred_mask_3c_epsilon

	final_blurred = masked_conv_image

	scipy.misc.toimage(d_image, cmin=0., cmax=255.).save(save_path + '{}_5_image.png'.format(i))
	scipy.misc.toimage(mask, cmin=0., cmax=1.).save(save_path + '{}_6_mask.png'.format(i))
	scipy.misc.toimage(blurred_image, cmin=0., cmax=255.).save(save_path + '{}_7_image_decomposed_blurred.png'.format(i))
	scipy.misc.toimage(blurred_mask, cmin=0., cmax=1.).save(save_path + '{}_8_disp_decomposed_binary_blurred.png'.format(i))
	scipy.misc.toimage(masked_conv_image, cmin=0., cmax=1.).save(save_path + '{}_9_masked_conv_image.png'.format(i))

	return masked_conv_image

def get_blurred_images(d_images, d_disp_binary_maps):
	masked_conv_images = []
	blurred_rings = []
	blurred_ring_masks = []

	masked_conv_image = np.zeros_like(d_images[0])
	for i in np.arange(num_layers):
		'''
		if i == 0:
			mask_stacked = d_disp_binary_maps[i]
		else:
			mask_stacked = np.sum(np.array(d_disp_binary_maps)[:i+1, :, :], axis = 0)
		'''

		#image_stacked = masked_conv_image + d_images[i]
		masked_conv_image = masked_gaussian_filter(d_images[i], d_disp_binary_maps[i], sigma[i], i)

		masked_conv_images.append(masked_conv_image)

	return masked_conv_images

def get_b_and_mconv_masks(mask, sigma):
	blurred_mask = gaussian_filter((mask).astype(float), (sigma, sigma))
	mconv_mask_epsilon = np.copy(blurred_mask)
	mconv_mask_epsilon[np.where(mconv_mask_epsilon == 0)] = 0.00001
	mconv_mask = np.copy(blurred_mask)
	mconv_mask = mconv_mask / mconv_mask_epsilon

	return blurred_mask, mconv_mask

def get_merged_images(original_image, m_images, masks):

	merged_image = np.zeros_like(m_images[0])
	mconv_mask_stacked_prev = np.zeros_like(masks[0])

	for i in np.arange(num_layers):
		if i == 0:
			merged_image = np.copy(m_images[i])
			scipy.misc.toimage(merged_image, cmin=0., cmax=1.).save(save_path + 'merge_{}_9_merged_image.png'.format(i))
			continue

		blurred_mask_stacked, mconv_mask_stacked = get_b_and_mconv_masks(1 - np.sum(np.array(masks)[:i], axis=0), sigma[i])
		blurred_mask, mconv_mask = get_b_and_mconv_masks(masks[i], sigma[i])
		scipy.misc.toimage(blurred_mask, cmin=0., cmax=1.).save(save_path + 'merge_{}_1_blurred_mask.png'.format(i))
		scipy.misc.toimage(mconv_mask, cmin=0., cmax=1.).save(save_path + 'merge_{}_2_mconv_mask.png'.format(i))
		scipy.misc.toimage(mconv_mask_stacked, cmin=0., cmax=1.).save(save_path + 'merge_{}_2_1_mconv_mask_stacked.png'.format(i))

		blurred_mask_prev, mconv_mask_prev = get_b_and_mconv_masks(masks[i - 1], sigma[i - 1])
		scipy.misc.toimage(mconv_mask_prev, cmin=0., cmax=1.).save(save_path + 'merge_{}_3_mconv_mask_prev.png'.format(i))

		mconv_mask_stacked_prev = mconv_mask_stacked_prev + mconv_mask_prev - (mconv_mask_stacked_prev * mconv_mask_prev)
		scipy.misc.toimage(mconv_mask_stacked_prev, cmin=0., cmax=1.).save(save_path + 'merge_{}_4_mconv_mask_stacked_prev.png'.format(i))

		################
		background = merged_image + original_image * expand_dims((1 - mconv_mask_stacked_prev))
		foreground = m_images[i] * expand_dims(mconv_mask_stacked) + merged_image * expand_dims(mconv_mask_stacked - mconv_mask)
		merge_mask = expand_dims(blurred_mask_stacked)
		merged_image = background * (1 - merge_mask) + foreground * merge_mask
		################

		scipy.misc.toimage(background, cmin=0., cmax=1.).save(save_path + 'merge_{}_7_background.png'.format(i))
		scipy.misc.toimage(foreground, cmin=0., cmax=1.).save(save_path + 'merge_{}_9_foreground.png'.format(i))
		scipy.misc.toimage(merge_mask, cmin=0., cmax=1.).save(save_path + 'merge_{}_10_merge_mask.png'.format(i))
		scipy.misc.toimage(merged_image, cmin=0., cmax=1.).save(save_path + 'merge_{}_11_merged_image.png'.format(i))

	return merged_image

def main():
	image = get_image(image_path, '0048.png')
	scipy.misc.toimage(image, cmin=0., cmax=255.).save(save_path + '00_image.png')
	disp_map = get_disparity_map(disp_map_path, '0048.pfm')
	disp_map_norm = ((disp_map / disp_map.max()) * 255).astype(int)

	#scipy.misc.toimage(image, cmin=0., cmax=255.).save(save_path + 'image.png')
	#scipy.misc.toimage(disp_map, cmin=0., cmax=disp_map.max()).save(save_path + 'disp.png')
	scipy.misc.toimage(disp_map_norm, cmin=0., cmax=255.).save(save_path + '00_disp_norm.png')

	decomposed_images, masks = get_decomposed_images(image, disp_map_norm, num_layers)
	masked_conv_images = get_blurred_images(decomposed_images, masks)
	merged_image = get_merged_images(image / 255., masked_conv_images, masks)
	scipy.misc.toimage(merged_image, cmin=0., cmax=1.).save(save_path + '00_merged_image.png')

if __name__ == '__main__':
    main()

