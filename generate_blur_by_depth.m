function generate_blur_by_depth(max_coc, is_random_gen, is_gpu, gpunum)
    if is_gpu
        disp(['gpu: ', num2str(gpunum)]);
        g = gpuDevice(gpunum);
    end

    % local all
    disp("reading files..")
    offset = 'D:';
    %offset = 'data1'
    image_file_paths = dir2([offset, filesep, 'synthetic_datasets', filesep, 'image', filesep, '**', filesep, '*']);
    depth_file_paths = dir2([offset, filesep, 'synthetic_datasets', filesep, 'depth', filesep, '**', filesep, '*']);
    kernel_file_paths = dir2([offset, filesep, 'kernel', filesep, '*']);
    disp("reading files.. DONE")

    %offset = ['result', filesep];
    offset = ['\\mango.postech.ac.kr\Users\JunyongLee\hub\datasets\SYNDOF\',num2str(max_coc),'_gaussian\'];
    dof_image_save_path = [offset, 'image', filesep];
    blur_map_save_path = [offset, 'blur_map', filesep];
    blur_map_norm_save_path = [offset, 'blur_map_norm', filesep];
    blur_map_binary_save_path = [offset, 'blur_map_binary', filesep];
    depth_map_save_path = [offset, 'depth_decomposed', filesep];

    offset
    mkdir(dof_image_save_path);
    mkdir(blur_map_save_path);
    mkdir(blur_map_norm_save_path);
    mkdir(blur_map_binary_save_path);
    mkdir(depth_map_save_path);

    num2generate = 10000;
    k = 0;
    while k <= num2generate
        rng('shuffle');
        for i = 1:length(image_file_paths)
            if is_random_gen
                select_idx = randi(length(image_file_paths));
            else
                select_idx = i;
            end
            image_file_path = char(image_file_paths(select_idx));
            depth_file_path = char(depth_file_paths(select_idx));
            fprintf('\n');
            disp([num2str(k), '/', num2str(num2generate)]);
            disp(['file_path: ', image_file_path]);
            disp(['depth_path: ', depth_file_path]);
            fprintf('\n');

            if contains(image_file_path, 'MPI') % synthia: in cm
                depth_scale = 1000.0;
            else
                depth_scale = 10;
            end

            disp('[blurring start..]')
            if is_gpu
                reset(g);
            end
            tic;
            %select_idx = randi(length(kernel_file_paths));
            %kernel_type = kernel_file_paths(select_idx);
            % only gaussian!
            kernel_type = 'gaussian';
            [dof_image, blur_map, blur_map_norm, blur_map_binary, depth_map, camera_params] = ... 
                blur_by_depth(image_file_path, depth_file_path, depth_scale, kernel_type, max_coc, is_gpu);
            toc;
            disp('[blurring DONE]')
            blur_map_temp = blur_map;
            blur_map_temp = blur_map_temp / 10.0;

            if max(blur_map_temp(:)) > (max_coc - 1 / 4.0)
                disp(num2str(max(blur_map_temp(:)), 2));
                disp('max coc is bigger than 30');
                continue;
            end

            [path, name, idx] = fileparts(image_file_path);
            if contains(image_file_path, 'SYNTHIA') % synthia: in cm
                prefix = 'SYNTHIA_';
            elseif contains(image_file_path, 'MPI')
                prefix = 'MPI_';
            else
                prefix = 'MIDDLEBURRY_';
            end
            path_prefix = split(path, filesep);
            path_prefix = cell2mat(path_prefix(end));
            prefix = [prefix, path_prefix, '_'];

            f = num2str(camera_params(1));
            fp = strrep(num2str(camera_params(2), 2), '.', '_');
            a = strrep(num2str(camera_params(3), 2), '.', '_');

            dof_image_save_name = [prefix, name, '_f_', f, '_fp_', fp, '_A_', a, '.png'];
            blur_map_save_name = [prefix, name, '_f_', f, '_fp_', fp, '_A_', a, '.png'];
            blur_map_norm_save_name = [prefix, name, '_f_', f, '_fp_', fp, '_A_', a, '_norm.png'];
            blur_map_binary_save_name = [prefix, name, '_f_', f, '_fp_', fp, '_A_', a, '_binary.png'];
            depth_map_name = [prefix, name, '_f_', f, '_fp_', fp, '_A_', a, '_depth_decomposed.png'];
            imwrite(dof_image, [dof_image_save_path, dof_image_save_name]);
            imwrite(blur_map, [blur_map_save_path, blur_map_save_name]);
            imwrite(blur_map_norm, [blur_map_norm_save_path, blur_map_norm_save_name]);
            imwrite(blur_map_binary, [blur_map_binary_save_path, blur_map_binary_save_name]);
            imwrite(depth_map, [depth_map_save_path, depth_map_name]);
            k = k + 1;
        end
    end
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
