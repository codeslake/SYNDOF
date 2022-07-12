function [blurred_image, blur_map_disc, blur_map_disc_norm, blur_map_binary, depth_map, camera_params] = blur_by_depth(image_file_name, depth_map_file_name, depth_scale, kernel_type, max_coc, is_gpu, g)
    % get image
    image = get_image(image_file_name);
    depth_map = get_depth_map(depth_map_file_name, depth_scale);

    max_depth_layers = 350;
    min_depth_layers = 200;

    depth_map_origin = depth_map;
    max_count = 20;
    count = 0;
    % compute CoCs
    while true
        [cocs, depth_map, camera_params] = get_cocs(depth_map_origin, size(image, 2), max_coc);

        unique_depth = flip(unique(depth_map))';
        if count > max_count
            break;
        end

        if length(unique_depth) < min_depth_layers || length(unique_depth) > max_depth_layers
            count = count + 1;
            continue;
        else
            break;
        end
    end

    % decompose images by unique depth (by cocs) and get masks
    [masks, images_decomposed] = get_decomposed_masks_and_images(image, depth_map, unique_depth);
    masks_sum_prev = get_masks_sum_prev(masks, cocs);

    % GPU
    if is_gpu
        masks = gpuArray(masks);
        images_decomposed = gpuArray(images_decomposed);
        masks_sum_prev = gpuArray(masks_sum_prev);
    end

    % print parameters
    print_params(camera_params, unique_depth, kernel_type, cocs);

    % Merging images
    fprintf('\n');
    disp('--------------------');
    disp('merging images..');
    merged_image = zeros(size(image));
    merged_mask = zeros(size(image));
    max_idx = size(masks, 3);
    coc_zero_idx = find(cocs <= 1);
    coc_zero_idx = coc_zero_idx(1);
    for i = 1:max_idx

        % get blurr components
        coc = cocs(i);
        [mask_blurred, image_blurred, blur_kernel] = get_blurred_components(masks(:, :, i), images_decomposed(i, :, :, :), coc, max_coc, kernel_type);

        % merge
        foreground = image_blurred;
        merge_mask = mask_blurred;

        % start inapint when focused plane already merged previously
        if i > coc_zero_idx
            [merged_image, merged_mask] = inpaint_background_and_modify_merged_mask(merged_image, merged_mask, masks_sum_prev(:, :, max_idx - i + 1), blur_kernel);
        end
        % merge
        merged_image = (1 - merge_mask) .* merged_image + foreground;
        merged_mask = (1 - merge_mask) .* merged_mask + merge_mask;
    end
    disp('merging images..DONE');
    disp('--------------------');

    % final blurred image
    blurred_image = merged_image ./ merged_mask;

    % final blur map
    [blur_map_disc, blur_map_disc_norm, blur_map_binary] = get_blur_map(masks, cocs, max_coc);
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
        reset(g);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function image = get_image(image_file_name)
    image = double(imread(image_file_name))/255.;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function depth_map = get_depth_map(depth_map_file_name, depth_scale)
    [~, ~, ext] = fileparts(depth_map_file_name);
    if strcmp(ext, '.dpt')
        depth_map = double(depth_read(depth_map_file_name)) * depth_scale;
    else
        depth_map = double(imread(depth_map_file_name)) * depth_scale;
    end
    if length(size(depth_map)) == 3
        depth_map = depth_map(:, :, 1);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cocs = compute_coc(z_point, z_focal, A, m_mmp, m_Cc)
    epsilon = 0.0001;
    coc_real = A .* (abs(z_point - z_focal)./(z_point + epsilon));
    cocs = double(m_mmp .* m_Cc .* coc_real);
    cocs = int64(cocs * 10);
    cocs = double(cocs) / 10.;
end

function [adjacent_unique, repeat_start_idx, propagation_idx] = adjacent_unique(array)
    adjacent_unique = array;
    adjacent_unique(diff(adjacent_unique) == 0) = [];

    repeat_start_idx = find(diff([array(1)-1 array]));
    propagation_idx = 1:length(repeat_start_idx);
    propagation_idx = repelem(propagation_idx, diff([0 find(diff(array)) numel(array)]));

end

function [unique_coc_image, depth_map] = get_unique_depth_map_and_coc_image(cocs, depth_map)

    [unique_coc_image, unique_coc_set_idx, unique_coc_propagate_idx] = adjacent_unique(cocs);

    diff_idx = 1:length(cocs);
    diff_idx(unique_coc_set_idx) = [];

    unique_depth = flip(unique(depth_map));
    depth_set_values = unique_depth(unique_coc_set_idx);
    depth_set_values = depth_set_values(unique_coc_propagate_idx);
    depth_set_values = depth_set_values(diff_idx);
    [depth_diff_idx, depth_set_values_idx] = ismember(depth_map, unique_depth(diff_idx));
    %depth_map(find(depth_diff_idx)) = depth_set_values(depth_set_values_idx(find(depth_diff_idx)));
    depth_map(depth_diff_idx > 0) = depth_set_values(depth_set_values_idx(depth_diff_idx > 0));
end


function [cocs, depth_map, camera_params] = get_cocs(depth_map, I_width, max_coc)
    %% camera parameters
    epsilon = 0.0001;

    % focal length
    f = datasample([50, 70, 85], 1, 'Weights', [0.1, 0.4, 0.4]);

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
    N = 1.0;
    A = f / N;

    % m_mmp
    sensor_size = 36; %[28.7, 36]mm
    m_mmp = I_width / sensor_size;

    % m_Cc
    m_Cc = z_view / z_focal;

    % (A)perture
    if compute_coc(max(z_point(:)), z_focal, A, m_mmp, m_Cc) > compute_coc(min(z_point(:)), z_focal, A, m_mmp, m_Cc)
        N = (f^2/(z_focal - f)) * (m_mmp / max_coc) * (abs(max(z_point(:)) - z_focal) / max(z_point(:)));
    else
        N = (f^2/(z_focal - f)) * (m_mmp / max_coc) * (abs(min(z_point(:)) - z_focal) / min(z_point(:)));
    end
    A = f / N;

    % compute CoC of an image
    cocs = compute_coc(z_point, z_focal, A, m_mmp, m_Cc);
    [cocs, depth_map] = get_unique_depth_map_and_coc_image(cocs, depth_map);

    % get camera params
    focal_idx = find(cocs == 0);
    focal_idx = focal_idx(1);
    z_focal_percent = 1 - double(focal_idx) / length(cocs);
    camera_params = [f, z_focal_percent, N];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function masks_sum_prev = get_masks_sum_prev(masks, cocs)
    coc_zero_idx = find(cocs <= 1);
    coc_zero_idx = coc_zero_idx(1);
    masks_sum_prev = cumsum(masks, 3);
    masks_sum_prev = masks_sum_prev(:, :, coc_zero_idx:end - 1);
    masks_sum_prev = flip(masks_sum_prev, 3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function blur_kernel = get_blur_kernel(coc, max_coc, kernel_type)
    if strcmp(kernel_type, 'gaussian')
        coc = double(coc);
        g_size = max_coc;
        s = (coc - 1) / 4.0;
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

        blur_kernel = imresize(blur_kernel, [coc * 2 + 1, coc * 2 + 1], 'bicubic');

        blur_kernel = blur_kernel ./ max(blur_kernel(:));
        blur_kernel = blur_kernel ./ sum(blur_kernel(:));
    end
end


function [mask_blurred, image_blurred, blur_kernel] = get_blurred_components(mask, image_decomposed, coc, max_coc, kernel_type)
    mask = double(mask);
    image_decomposed = squeeze(image_decomposed);
    %% blur image and mask
    if coc > 1
        blur_kernel = get_blur_kernel(coc, max_coc, kernel_type);

        concated_mask_image = cat(3, mask, image_decomposed);
        concated_blurred_mask_image = imfilter(concated_mask_image, blur_kernel, 'same', 'conv', 'symmetric');

        mask_blurred = concated_blurred_mask_image(:, :, 1);
        image_blurred = concated_blurred_mask_image(:, :, 2:4);
    else
        mask_blurred = mask;
        image_blurred = image_decomposed;
        blur_kernel = 0;
    end

    mask_blurred(mask_blurred < 0.0001) = 0;
    mask_blurred = repelem(mask_blurred, 1, 1, 3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [merged_image, merged_mask] = inpaint_background_and_modify_merged_mask(merged_image, merged_mask, mask_sum_prev, blur_kernel)
    % blur cumulated masks before curret mask with current blur kernel
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [blur_map, blur_map_norm, blur_map_binary] = get_blur_map(masks, cocs, max_coc)
    blur_map = sum(double(masks) .* reshape(cocs, 1, 1, length(cocs)), 3);
    blur_map(blur_map < 1) = 1;

    blur_map_binary = blur_map; 
    blur_map_binary(blur_map_binary <= 1) = 0;
    blur_map_binary(blur_map_binary > 1) = 1;

    blur_map = (blur_map - 1.) ./ 4.0; % converting from coc to sigma
    blur_map_norm = blur_map - min(blur_map(:));
    blur_map_norm = blur_map_norm ./ max(blur_map_norm(:));
    
    blur_map = uint16(blur_map .* 10.0);
    blur_map = repelem(blur_map, 1, 1, 3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function print_params(camera_params, unique_depth, kernel_type, cocs)
    disp('-----------------');
    disp('camera parameters');
    disp('-----------------');
    disp(['focal length: ', num2str(camera_params(1)), 'mm']);
    disp(['focal point: ', num2str(camera_params(2), 3)]);
    disp(['aperture number: ', num2str(camera_params(3), 3)]);

    disp('------------');
    disp('merging info');
    disp('------------');
    disp(['depth layers: ', num2str(length(unique_depth))]);
    disp(['aperture shape: ', char(kernel_type)]);
    disp(['max coc: ', num2str(max(cocs(:)))]);
end
