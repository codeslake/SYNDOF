function [blurred_image, blur_map, blur_map_norm, blur_map_disc, blur_map_disc_norm, blur_map_binary, depth_map, camera_params] = blur_by_depth(image_file_name, depth_map_file_name, depth_scale, kernel_type, is_gpu)
%function [blurred_image, blur_map, blur_map_norm, camera_params] = blur_by_depth()
    %close all;
    %clear;
    %delete 'output/merged/*.png'
    %delete 'output/result/*.png'

    %% for test
    %image_file_name = '/data1/synthetic_datasets/image/SYNTHIA/RAND_CITYSCAPES/0001864.png';
    %depth_map_file_name = '/data1/synthetic_datasets/depth/SYNTHIA/RAND_CITYSCAPES/0001864.png';
    %depth_scale = 10.0;
    %is_gpu = false;
    %%

    %% get image
    image = get_image(image_file_name);
    depth_map = get_depth_map(depth_map_file_name, depth_scale);

    %% get unique depth in max_depth range
    %[max_unique_coc_image_radius, null] = get_coc_image_radius(unique(depth_map), size(image, 2), true);
    %max_depth_step = min(max_unique_coc_image_radius, 300);
    %if max_depth_step == 300
        %depth_map = get_unique_depth_map(depth_map, max_depth_step);
        %[coc_radiuses, camera_params] = get_coc_image_radius(depth_map, size(image, 2));
    %else
    [coc_radiuses, camera_params, depth_map, blur_map] = get_coc_image_radius(depth_map, size(image, 2));
    [coc_radiuses, depth_map] = get_unique_depth_map_and_coc_image_radius(depth_map, coc_radiuses, size(image, 2));
    blur_map_norm = blur_map - min(blur_map(:));
    blur_map_norm = blur_map_norm ./ max(blur_map_norm(:));
    blur_map = uint16(blur_map .* 10.0);
    blur_map = repelem(blur_map, 1, 1, 3);
    %end
    %% decomposing images and masks
    unique_depth = flip(unique(depth_map))';
    [masks, images_decomposed] = get_decomposed_masks_and_images(image, depth_map, unique_depth);
    masks_sum_prev = get_masks_sum_prev(masks, coc_radiuses);

    %%GPU
    if is_gpu
        masks = gpuArray(masks);
        images_decomposed = gpuArray(images_decomposed);
        masks_sum_prev = gpuArray(masks_sum_prev);
    end

    disp(['-----------------']);
    disp(['camera parameters']);
    disp(['-----------------']);
    disp(['focal length: ', num2str(camera_params(1)), 'mm']);
    disp(['focal point: ', num2str(camera_params(2), 3)]);
    disp(['aperture number: ', num2str(camera_params(3), 3)]);

    disp(['------------']);
    disp(['merging info']);
    disp(['------------']);
    disp(['depth layers: ', num2str(length(unique_depth))]);
    disp(['aperture shape: ', char(kernel_type)]);
    disp(['max coc radius: ', num2str(max(coc_radiuses(:)))]);

    %% Merging images
    fprintf('\n');
    disp(['--------------------']);
    disp(['merging images..']);
    merged_image = zeros(size(image));
    merged_mask = zeros(size(image));
    inpaint_flag = false;
    max_idx = size(masks, 3);
    for i = 1:max_idx

        %% get blurr components
        coc_radius = coc_radiuses(i);
        [mask_blurred, image_blurred, blur_kernel] = get_blurred_components(masks(:, :, i), images_decomposed(i, :, :, :), coc_radius, kernel_type);

        %%merge
        foreground = image_blurred;
        merge_mask = mask_blurred;

        % inapint when inpainting flag is true
        if inpaint_flag == true
            mask_sum_prev = logical(sum(masks(:, :, 1:i - 1), 3));
            [merged_image, merged_mask] = inpaint_background_and_modify_merged_mask(merged_image, merged_mask, masks_sum_prev(:, :, max_idx - i + 1), blur_kernel);
        end
        % merge
        merged_image = (1 - merge_mask) .* merged_image + foreground;
        merged_mask = (1 - merge_mask) .* merged_mask + merge_mask;
        % set inapinting flag
        inpaint_flag = toggle_inpaint_flag(inpaint_flag, coc_radius);
        merged_mask_temp = merged_mask;
        merged_mask_temp(merged_mask_temp == 0) = 1;
        %imwrite(gather(merged_image ./ merged_mask_temp), ['output/merged/', num2str(i), '_merged_image_norm.png']);
    end
    disp(['merging images..DONE']);
    disp(['--------------------']);

    %% final blurred image
    blurred_image = merged_image ./ merged_mask;

    %% final blur map
    [blur_map_disc, blur_map_disc_norm, blur_map_binary] = get_blur_map(masks, coc_radiuses);
    % imwrite(gather(image), ['output/merged/', 'original.png']);
    % imwrite(gather(merged_image), ['output/merged/', 'final_merged_image.png']);
    % imwrite(gather(blur_map_disc), ['output/merged/', 'final_blur_map.png']);
    % imwrite(rescale(gather(blur_map_disc_norm)), ['output/merged/', 'final_blur_map_disc_norm.png']);
    depth_map = depth_map - min(depth_map(:));
    depth_map = depth_map ./ max(depth_map(:));
    depth_map = repelem(depth_map, 1, 1, 3);
    depth_map = uint8(depth_map .* 255);

    if is_gpu
        blurred_image = gather(blurred_image);
        blur_map_disc = gather(blur_map_disc);
        blur_map_disc_norm = gather(blur_map_disc_norm);
        blur_map_binary = gather(blur_map_binary);
        depth_map = gather(depth_map);
    end
end

function image = get_image(image_file_name)
    image = double(imread(image_file_name))/255.;
end

function depth_map = get_depth_map(depth_map_file_name, depth_scale)
    [path, name, ext] = fileparts(depth_map_file_name);
    if strcmp(ext, '.dpt')
        depth_map = double(depth_read(depth_map_file_name)) * depth_scale;
    else
        depth_map = double(imread(depth_map_file_name)) * depth_scale;
    end
    if length(size(depth_map)) == 3
        depth_map = depth_map(:, :, 1);
    end
end

function depth_map = get_unique_depth_map(depth_map, max_depth_step)

    unique_depth = unique(depth_map);
    %[unique_probs, ] = histcounts(depth_map, [unique_depth; unique_depth(end)], 'Normalization', 'probability');
    [unique_probs, ] = histcounts(depth_map, [unique_depth; unique_depth(end)]);
    max_depth_step = min(max_depth_step, length(unique_depth));

    depth_step = 0;
    current_prob = 0.0;
    %remaining_prob = 1.0;
    remaining_prob = sum(unique_probs(:));
    marked_depth = zeros(size(unique_probs));
    marking_idx = 1;

    [sorted_unique_probs, I] = sort(unique_probs, 'descend');
    sorted_unique_probs_idx = 1:length(unique_probs);
    sorted_unique_probs_idx = sorted_unique_probs_idx(I);
    for i = 1:length(unique_depth)
        unique_prob = sorted_unique_probs(i);
        %max_prob = remaining_prob / (max_depth_step - depth_step);
        max_prob = uint64(linspace(1, remaining_prob, max_depth_step - depth_step));
        max_prob = max_prob(2);
        if unique_prob >= max_prob
            depth_step = depth_step + 1;
            remaining_prob = remaining_prob - unique_prob;
            marked_depth(sorted_unique_probs_idx(i)) = 1;
        end
    end

    for i = 1:length(unique_depth)
        unique_prob = unique_probs(i);
        %max_prob = remaining_prob / (max_depth_step - depth_step);
        max_prob = uint64(linspace(1, remaining_prob, max_depth_step - depth_step));
        if length(max_prob) == 0
            depth_map(find(depth_map > low & depth_map <= unique_depth(i))) = unique_depth(i);
            break;
        elseif length(max_prob) == 1
            max_prob = max_prob(1);
        else
            max_prob = max_prob(2);
        end

        if current_prob + unique_prob >= max_prob
            if i == 1
                low = 0;
            else
                low = unique_depth(marking_idx);
            end
            if marked_depth(i) == 1;
                if current_prob >= max_prob
                    depth_map(find(depth_map > low & depth_map <= unique_depth(i - 1))) = unique_depth(i - 1);
                    marking_idx = i - 1;
                    low = unique_depth(marking_idx);
                    depth_step = depth_step + 1;
                end
                remaining_prob = remaining_prob - current_prob;
                depth_map(find(depth_map > low & depth_map <= unique_depth(i))) = unique_depth(i);
                marking_idx = i;
            else
                depth_map(find(depth_map > low & depth_map <= unique_depth(i))) = unique_depth(i);
                remaining_prob = remaining_prob - (current_prob + unique_prob);
                marking_idx = i;
                depth_step = depth_step + 1;
            end
            current_prob = 0.0;
        else
            current_prob = current_prob + unique_prob;
        end
    end
end

function coc = compute_coc(z_point, z_focal, A, m_mmp, m_Cc)
    epsilon = 0.0001;
    coc_real_diameter = A .* (abs(z_point - z_focal)./(z_point + epsilon));
    coc_image_diameter = m_mmp .* m_Cc .* coc_real_diameter;
    coc_image_diameter = floor(coc_image_diameter);
    coc = coc_real_diameter;
end

function [coc_image_radius, camera_params, depth_map, blur_map] = get_coc_image_radius(depth_map, I_width, is_max)
    %% camera parameters
    epsilon = 0.0001;

    % focal length
    %f = 70; %[14, 20, 24, 35, 50, 70, 85]mm
    if nargin >= 3
        f = max([50, 70, 85]);
    else
        f = datasample([50, 70, 85], 1, 'Weights', [0.1, 0.4, 0.4]);
    end

    depth_scale_factor = 5.0;
    if min(depth_map(:)) < depth_scale_factor * f
        disp('z_near equal or less than focal length. Shifting the depth..');
        depth_map = depth_map - min(depth_map(:));
        depth_map = depth_map + depth_scale_factor * f;
    end
    z_point = double(flip(unique(depth_map)));
    z_point = z_point';

    select_weight = (length(z_point) / 2.) * randn([1, 10000]) + length(z_point);
    [select_weight,] = histcounts(select_weight, length(z_point) * 2);
    select_weight = select_weight(1:length(z_point));
    select_weight = select_weight ./ sum(select_weight(:));
    z_focal = datasample(z_point, 1, 'Weights', select_weight);

    z_view = (f * z_focal) / (z_focal - f + epsilon);

    % lens diameter
    % aperture numbers [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11, 16, 22]mm
    N = 1.0;
    A = f / N;

    % m_mmp
    sensor_size = 36; %[28.7, 36]mm
    m_mmp = I_width / sensor_size;

    % m_Cc
    m_Cc = z_view / z_focal;

    if compute_coc(max(z_point(:)), z_focal, A, m_mmp, m_Cc) > compute_coc(min(z_point(:)), z_focal, A, m_mmp, m_Cc)
        N = (f^2/(z_focal - f)) * (m_mmp / 30) * (abs(max(z_point(:)) - z_focal) / max(z_point(:)));
    else
        N = (f^2/(z_focal - f)) * (m_mmp / 30) * (abs(min(z_point(:)) - z_focal) / min(z_point(:)));
    end
    % temp_N =[1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11, 16, 22]; 
    % diff_N = abs(temp_N - N);
    % diff_N = find(diff_N == min(diff_N(:)));
    % N = temp_N(diff_N(1));
    A = f / N;

    coc_real_diameter = A .* (abs(z_point - z_focal)./(z_point + epsilon));
    coc_image_diameter = m_mmp .* m_Cc .* coc_real_diameter;
    coc_image_diameter = floor(coc_image_diameter);
    coc_image_diameter(find(mod(coc_image_diameter, 2) == 0 & coc_image_diameter ~= 0)) = coc_image_diameter(find(mod(coc_image_diameter, 2) == 0 & coc_image_diameter ~= 0)) - 1;
    coc_image_radius = floor(coc_image_diameter ./ 2.0);

    blur_map = double(depth_map);
    blur_map = A .* (abs(blur_map - z_focal)./(blur_map + epsilon));
    blur_map = m_mmp .* m_Cc .* blur_map;

    focal_idx = find(coc_image_radius == 0);
    focal_idx = focal_idx(1);
    z_focal_percent = 1 - double(focal_idx) / length(coc_image_radius);
    camera_params = [f, z_focal_percent, N];

    if nargin >= 3
        [coc_image_radius, ~, ~] = adjacent_unique(coc_image_radius);
        coc_image_radius = length(coc_image_radius);
    end
end

function [unique_coc_image_radius, depth_map] = get_unique_depth_map_and_coc_image_radius(depth_map, coc_image_radius, I_width)

    [unique_coc_image_radius, unique_coc_set_idx, unique_coc_propagate_idx] = adjacent_unique(coc_image_radius);

    diff_idx = [1:length(coc_image_radius)];
    diff_idx(unique_coc_set_idx) = [];

    unique_depth = flip(unique(depth_map));
    depth_set_values = unique_depth(unique_coc_set_idx);
    depth_set_values = depth_set_values(unique_coc_propagate_idx);
    depth_set_values = depth_set_values(diff_idx);
    [depth_diff_idx, depth_set_values_idx] = ismember(depth_map, unique_depth(diff_idx));
    depth_map(find(depth_diff_idx)) = depth_set_values(depth_set_values_idx(find(depth_diff_idx)));
end

function [masks, images_decomposed] = get_decomposed_masks_and_images(image, depth_map, unique_depth)
    depth_step = length(unique_depth);
    depth_map_decomposed = repmat(depth_map, [1, 1, depth_step]);

    masks = logical(depth_map_decomposed == reshape(unique_depth, 1, 1, depth_step));
    %hist_depth_map_decomposed = sum(sum(masks, 1), 2)

    images_decomposed = repmat(image, [1, 1, depth_step]);
    shape = size(images_decomposed);
    images_decomposed = reshape(images_decomposed, shape(1), shape(2), 3, depth_step);
    images_decomposed = permute(images_decomposed, [4, 1, 2, 3]);

    masks_3c = repelem(masks, 1, 1, 3);
    shape = size(masks_3c);
    masks_3c = reshape(masks_3c, shape(1), shape(2), 3, depth_step);
    masks_3c = permute(masks_3c, [4, 1, 2, 3]);
    images_decomposed = images_decomposed .* masks_3c;
end

function masks_sum_prev = get_masks_sum_prev(masks, coc_radiuses)
    coc_zero_idx = find(coc_radiuses == 0);
    shape = size(masks);
    masks_sum_prev = cumsum(masks, 3);
    masks_sum_prev = masks_sum_prev(:, :, coc_zero_idx:end - 1);
    masks_sum_prev = flip(masks_sum_prev, 3);
end

function [mask_blurred, image_blurred, blur_kernel] = get_blurred_components(mask, image_decomposed, coc_radius, kernel_type)
    mask = double(mask);
    image_decomposed = squeeze(image_decomposed);
    %% blur image and mask
    if coc_radius ~= 0
        blur_kernel = get_blur_kernel(coc_radius, kernel_type);

        concated_mask_image = cat(3, mask, image_decomposed);
        concated_blurred_mask_image = imfilter(concated_mask_image, blur_kernel, 'same', 'conv', 'symmetric');

        mask_blurred = concated_blurred_mask_image(:, :, 1);
        image_blurred = concated_blurred_mask_image(:, :, 2:4);
    else
        mask_blurred = mask;
        image_blurred = image_decomposed;
        blur_kernel = 0;
    end

    mask_blurred(mask_blurred < 0.001) = 0;
    mask_blurred = repelem(mask_blurred, 1, 1, 3);
end

function blur_kernel = get_blur_kernel(coc_radius, kernel_type)
    if strcmp(kernel_type, 'gaussian')
        s = double(coc_radius);
        g_size = double(2 * ceil(10 * s) + 1);
        G = fspecial('gaussian',[g_size, g_size], s);
        G_idx = find(G > 0);
        [y,x] = ind2sub(size(G),G_idx);
        blur_kernel = G(min(y):max(y), min(x):max(x));
    else
        blur_kernel = double(imread(char(kernel_type))) / 255.;

        if length(size(blur_kernel)) == 3
            blur_kernel = blur_kernel(:, :, 1);
        end
        blur_kernel_idx = find(blur_kernel > 0);
        [y,x] = ind2sub(size(blur_kernel),blur_kernel_idx);
        blur_kernel = blur_kernel(min(y):max(y), min(x):max(x));

        blur_kernel = imresize(blur_kernel, [coc_radius * 2 + 1, coc_radius * 2 + 1], 'bicubic');

        blur_kernel = blur_kernel ./ max(blur_kernel(:));
        blur_kernel = blur_kernel ./ sum(blur_kernel(:));
    end
end

function [merged_image, merged_mask] = inpaint_background_and_modify_merged_mask(merged_image, merged_mask, mask_sum_prev, blur_kernel);
    % blur cumlated masks before curret mask with current blur kernel
    mask_sum_prev_blurred = imfilter(double(mask_sum_prev), blur_kernel, 'same', 'conv', 'symmetric');
    mask_sum_prev_blurred(mask_sum_prev_blurred < 0.001) = 0;


    % get mask to inpaint : area that previous masks affects current mask
    inpainting_mask = mask_sum_prev_blurred;
    mask_sum_prev_blurred(mask_sum_prev_blurred == 0) = 0.00001;
    inpainting_mask = inpainting_mask ./ mask_sum_prev_blurred;
    inpainting_mask(mask_sum_prev == 1) = 0;
    inpainting_mask = logical(inpainting_mask);

    % inpaint image with mask to inpaint
    mask_sum_prev_3c = logical(repelem(mask_sum_prev, 1, 1, 3));
    merged_mask_temp = merged_mask;
    merged_mask_temp(merged_mask_temp == 0) = 1;
    image_inpainted = merged_image ./ merged_mask_temp;
    image_inpainted(mask_sum_prev_3c == 0) = 0;
    for j = 1:3
        image_inpainted(:, :, j) = regionfill(gather(image_inpainted(:, :, j)), gather(inpainting_mask));
    end
    % inpaint mask
    mask_inpainted = regionfill(gather(double(mask_sum_prev)), gather(inpainting_mask));
    mask_inpainted = repelem(mask_inpainted, 1, 1, 3);

    % modify merged image and merged_mask
    inpainting_mask = repelem(inpainting_mask, 1, 1, 3);
    merged_mask_temp = merged_mask;
    merged_mask_temp(merged_mask_temp == 0) = 1;
    merged_image = merged_image ./ merged_mask_temp;
    merged_mask = merged_mask ./ merged_mask_temp;

    merged_image(inpainting_mask) = image_inpainted(inpainting_mask);
    merged_mask(inpainting_mask) = mask_inpainted(inpainting_mask);
end

function inpaint_flag = toggle_inpaint_flag(inpaint_flag, coc_radius)
    if inpaint_flag == false
        if coc_radius ~= 0
            inpaint_flag = false;
        else
            inpaint_flag = true;
        end
    end
end

function [blur_map, blur_map_norm, blur_map_binary] = get_blur_map(masks, coc_radiuses)
    blur_map = sum(double(masks) .* reshape(coc_radiuses * 2 + 1, 1, 1, length(coc_radiuses)), 3);
    blur_map_norm = blur_map - min(blur_map(:));
    blur_map_norm = blur_map_norm ./ max(blur_map_norm(:));
    blur_map_binary = blur_map; 
    blur_map_binary(blur_map_binary <= 1) = 0;
    blur_map_binary(blur_map_binary > 1) = 1;

    % % normalizing between 0 and 1 possible max of coc
    % blur_map = blur_map ./ 755;
    % if max(blur_map(:)) >= 1
    %     disp(['normalizing error!'])
    %     blur_map = blur_map ./ max(blur_map(:));
    % end

    blur_map = uint16(blur_map .* 10.0);
    blur_map = repelem(blur_map, 1, 1, 3);
end

function full_path = dir2(varargin)
    if nargin == 0
        name = '.';
    elseif nargin == 1
        name = varargin{1};
    else
        error('Too many input arguments.')
    end

    listing = dir(name);

    inds = [];
    n    = 0;
    k    = 1;

    while k <= length(listing)
        if listing(k).isdir
            inds(end + 1) = k;
        end
        k = k + 1;
    end
    listing(inds) = [];

    full_path = [];
    for k = 1:length(listing)
        file_path = listing(k).folder;
        file_name = listing(k).name;
        full_path = [full_path, string(fullfile(file_path, file_name))];
    end
    full_path = sort(full_path);
end

function [adjacent_unique, repeat_start_idx, propagation_idx] = adjacent_unique(array)
    adjacent_unique = array;
    adjacent_unique(diff(adjacent_unique) == 0) = [];

    repeat_start_idx = find(diff([array(1)-1 array]));
    propagation_idx = 1:length(repeat_start_idx);
    propagation_idx = repelem(propagation_idx, diff([0 find(diff(array)) numel(array)]));

end
