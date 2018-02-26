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
sigma = [9, 9, 3, 0]

def get_image(path, file_name):
    return scipy.misc.imread(path + file_name, mode='RGB')

def get_disparity_map(path, file_name):
	pfm_file = open(path + file_name, 'r')
	return pfm_handler.load_pfm(pfm_file)

def get_decomposed_images(image, disp_map, num_layers):
	interval_distance = 256 / num_layers
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
		#decomposed_disp_binary = decomposed_disp_binary.astype(float)
		#decomposed_disp_binary[np.where(decomposed_disp_binary == 0)] += 0.00001
		decomposed_disp_maps_binary.append(decomposed_disp_binary)

		scipy.misc.toimage(decomposed_image, cmin=0., cmax=255.).save(save_path + '{}_image_decomposed.png'.format(i))
		scipy.misc.toimage(decomposed_disp, cmin=0., cmax=255.).save(save_path + '{}_disp_decomposed.png'.format(i))
		scipy.misc.toimage(decomposed_disp_binary, cmin=0., cmax=1.).save(save_path + '{}_disp_decomposed_binary.png'.format(i))
		scipy.misc.toimage(1-decomposed_disp_binary, cmin=0., cmax=1.).save(save_path + '{}_disp_decomposed_binary_flip.png'.format(i))

	return decomposed_images, decomposed_disp_maps_binary

def masked_gaussian_filter(d_image, mask, sigma, i):
	blurred_image = gaussian_filter(d_image, (sigma, sigma, 0))
	blurred_mask = gaussian_filter(mask.astype(float), (sigma, sigma))

	blurred_mask_3c = np.repeat(np.expand_dims(blurred_mask, 2), 3, axis = 2)
	mask_3c = np.repeat(np.expand_dims(mask, 2), 3, axis = 2)
	blurred_mask_3c_epsilon = blurred_mask_3c
	blurred_mask_3c_epsilon[np.where(blurred_mask_3c_epsilon == 0)] = 0.00001

	masked_conv_image = blurred_image / blurred_mask_3c_epsilon * mask_3c
	blurred_ring = (1 - mask_3c) * blurred_image
	blurred_ring_mask = (1 - mask) * blurred_mask 
	blurred_ring_mask[np.where(blurred_ring_mask > 0.00001)] = 1 
	blurred_ring_mask_3c = np.repeat(np.expand_dims(blurred_ring_mask, 3), 3, axis = 2)
	final_blurred = masked_conv_image + blurred_ring

	scipy.misc.toimage(blurred_image, cmin=0., cmax=255.).save(save_path + '{}_image_decomposed_blurred.png'.format(i))
	scipy.misc.toimage(blurred_mask, cmin=0., cmax=1.).save(save_path + '{}_disp_decomposed_binary_blurred.png'.format(i))
	scipy.misc.toimage(final_blurred, cmin=0., cmax=255.).save(save_path + '{}_final_blurred.png'.format(i))
	scipy.misc.toimage(masked_conv_image, cmin=0., cmax=255.).save(save_path + '{}_maksed_conv_image.png'.format(i))
	scipy.misc.toimage(blurred_ring, cmin=0., cmax=255.).save(save_path + '{}_ring_blurred.png'.format(i))
	scipy.misc.toimage(blurred_ring_mask, cmin=0., cmax=1.).save(save_path + '{}_ring_blurred_mask.png'.format(i))

	return masked_conv_image, mask_3c, blurred_ring, blurred_ring_mask_3c

def get_blurred_images(d_images, d_disp_binary_maps):
	masked_conv_images = []
	masks = []
	blurred_rings = []
	blurred_ring_masks = []

	for i in np.arange(num_layers):
	    masked_conv_image, mask, blurred_ring, blurred_ring_mask = masked_gaussian_filter(d_images[i], d_disp_binary_maps[i], sigma[i], i)

	    masked_conv_images.append(masked_conv_image)
	    masks.append(mask)
	    blurred_rings.append(blurred_ring)
	    blurred_ring_masks.append(blurred_ring_mask)

	return masked_conv_images, masks, blurred_rings, blurred_ring_masks

def get_merged_images(m_images, masks, b_rings, b_ring_masks):
	#merged_image = np.sum(np.array(b_images), axis=0)
	# 1. weighted sum of rings
	#sum_rings = np.sum(rings).astype(float) / num_layers
	# 2. stack (decomposed images - ring) by depth
	merged_image = np.zeros_like(m_images[0])
	for i in np.arange(num_layers):
		if i == 0:
			alpha = 0
		else:
			alpha = 0.5

		merged_image = merged_image * (1 - masks[i]) + m_images[i]
		scipy.misc.toimage(merged_image, cmin=0., cmax=255.).save(save_path + 'test_{}_1_merged_image.png'.format(i))

		factor = (merged_image * b_ring_masks[i])  + b_rings[i]
		factor[np.where(factor == 0)] = 0.00001
		background = (merged_image * b_ring_masks[i]) * (merged_image * b_ring_masks[i]) / factor
		foreground = b_rings[i] * b_rings[i] / factor
		blended_ring = (background + foreground) * b_ring_masks[i]

		scipy.misc.toimage(factor, cmin=0., cmax=255.).save(save_path + 'test_{}_factor.png'.format(i))
		scipy.misc.toimage(background, cmin=0., cmax=255.).save(save_path + 'test_{}_background.png'.format(i))
		scipy.misc.toimage(foreground, cmin=0., cmax=255.).save(save_path + 'test_{}_foreground.png'.format(i))
		scipy.misc.toimage(blended_ring, cmin=0., cmax=255.).save(save_path + 'test_{}_2_blended_ring.png'.format(i))

		merged_image = merged_image * (1 - b_ring_masks[i]) + blended_ring
		scipy.misc.toimage(b_ring_masks[i], cmin=0., cmax=1.).save(save_path + 'test_{}_3_b_ring_mask.png'.format(i))
		scipy.misc.toimage(1-b_ring_masks[i], cmin=0., cmax=1.).save(save_path + 'test_{}_4_b_ring_mask_flipped.png'.format(i))

		#merged_ring = (alpha * (merged_image * b_ring_masks[i])  + (1 - alpha) * b_rings[i]) 
		scipy.misc.toimage(merged_image, cmin=0., cmax=255.).save(save_path + 'merged_image_{}.png'.format(i))
		#scipy.misc.toimage(merged_ring, cmin=0., cmax=255.).save(save_path + 'merged_ring_{}.png'.format(i))

	return merged_image

def main():
	image = get_image(image_path, '0048.png')
	scipy.misc.toimage(image, cmin=0., cmax=255.).save(save_path + 'image.png')
	disp_map = get_disparity_map(disp_map_path, '0048.pfm')
	disp_map_norm = ((disp_map / disp_map.max()) * 255).astype(int)

	#scipy.misc.toimage(image, cmin=0., cmax=255.).save(save_path + 'image.png')
	#scipy.misc.toimage(disp_map, cmin=0., cmax=disp_map.max()).save(save_path + 'disp.png')
	scipy.misc.toimage(disp_map_norm, cmin=0., cmax=255.).save(save_path + 'disp_norm.png')

	decomposed_images, decomposed_disp_maps_binary = get_decomposed_images(image, disp_map_norm, num_layers)
	masked_conv_images, masks, blurred_rings, blurred_ring_masks = get_blurred_images(decomposed_images, decomposed_disp_maps_binary)
	merged_image = get_merged_images(masked_conv_images, masks, blurred_rings, blurred_ring_masks)
	scipy.misc.toimage(merged_image, cmin=0., cmax=255.).save(save_path + 'merged_image.png')

if __name__ == '__main__':
    main()

