function export = export_matlab_rt_json(output_path, tx_sites, rx_sites, propagation_model, varargin)
%EXPORT_MATLAB_RT_JSON Export MATLAB Ray Tracing results for DeepMIMO.
%
%   export = EXPORT_MATLAB_RT_JSON(output_path, tx_sites, rx_sites, propagation_model)
%   runs RAYTRACE(tx_sites, rx_sites, propagation_model) and writes a JSON
%   file that can be converted with DeepMIMO's convert_matlab_rt_json(...).
%
%   export = EXPORT_MATLAB_RT_JSON(..., "Rays", rays_cell) uses precomputed
%   raytrace output instead of running raytrace again.
%
%   Optional name-value arguments:
%       "Experiment"  - metadata experiment name
%       "Description" - metadata description
%       "Scene"       - scalar struct merged into the exported scene block
%
%   The export always uses the multi-link JSON schema. Every TX/RX pair is
%   written explicitly; links without rays are exported with num_rays = 0 and
%   rays = [].

    parser = inputParser;
    parser.FunctionName = "export_matlab_rt_json";
    parser.addParameter("Rays", [], @(value) true);
    parser.addParameter("Experiment", "matlab_rt_export", @is_text_scalar);
    parser.addParameter("Description", "MATLAB Ray Tracing JSON export", @is_text_scalar);
    parser.addParameter("Scene", struct(), @is_scalar_struct);
    parser.parse(varargin{:});
    options = parser.Results;

    tx_sites = tx_sites(:).';
    rx_sites = rx_sites(:).';
    num_tx = numel(tx_sites);
    num_rx = numel(rx_sites);

    if num_tx == 0 || num_rx == 0
        error("export_matlab_rt_json:EmptySites", ...
            "At least one TX site and one RX site are required.");
    end

    if isempty(options.Rays)
        rays_output = raytrace(tx_sites, rx_sites, propagation_model);
    else
        rays_output = options.Rays;
    end
    rays_cell = normalize_rays_cell(rays_output, num_tx, num_rx);

    export = struct();
    export.metadata = struct( ...
        "experiment", char(string(options.Experiment)), ...
        "matlab_version", version, ...
        "description", char(string(options.Description)));
    export.scene = build_scene(tx_sites, rx_sites, options.Scene);
    export.propagation_model = export_propagation_model(propagation_model);
    export.num_tx = num_tx;
    export.num_rx = num_rx;
    export.transmitters = export_transmitters(tx_sites);
    export.receivers = export_receivers(rx_sites);
    export.links = export_links(tx_sites, rx_sites, rays_cell);

    json_text = encode_json(export);
    write_text(output_path, json_text);
end

function scene = build_scene(tx_sites, rx_sites, scene_overrides)
    scene = struct( ...
        "coordinate_system", "cartesian", ...
        "frequency_hz", site_frequency_hz(tx_sites), ...
        "tx_positions_m", site_positions_m(tx_sites), ...
        "rx_positions_m", site_positions_m(rx_sites), ...
        "geometry", "not exported; DeepMIMO converter writes an empty scene placeholder");
    scene = merge_structs(scene, scene_overrides);
end

function records = export_transmitters(tx_sites)
    records = cell(1, numel(tx_sites));
    for idx = 1:numel(tx_sites)
        site = tx_sites(idx);
        records{idx} = struct( ...
            "index", idx, ...
            "class", class(site), ...
            "name", string_value(get_prop(site, "Name", sprintf("tx%d", idx))), ...
            "antenna_position_m", vector3(get_prop(site, "AntennaPosition", [])), ...
            "transmitter_frequency_hz", scalar_value( ...
                get_prop(site, "TransmitterFrequency", []), ...
                "txsite.TransmitterFrequency"));
    end
end

function records = export_receivers(rx_sites)
    records = cell(1, numel(rx_sites));
    for idx = 1:numel(rx_sites)
        site = rx_sites(idx);
        records{idx} = struct( ...
            "index", idx, ...
            "class", class(site), ...
            "name", string_value(get_prop(site, "Name", sprintf("rx%d", idx))), ...
            "antenna_position_m", vector3(get_prop(site, "AntennaPosition", [])));
    end
end

function records = export_links(tx_sites, rx_sites, rays_cell)
    num_tx = numel(tx_sites);
    num_rx = numel(rx_sites);
    records = cell(1, num_tx * num_rx);
    link_index = 1;

    for tx_idx = 1:num_tx
        for rx_idx = 1:num_rx
            tx = tx_sites(tx_idx);
            rx = rx_sites(rx_idx);
            ray_list = normalize_ray_list(rays_cell{tx_idx, rx_idx});
            ray_records = cell(1, numel(ray_list));
            for ray_idx = 1:numel(ray_list)
                ray_records{ray_idx} = export_ray(ray_list(ray_idx), ray_idx);
            end

            records{link_index} = struct( ...
                "index", link_index, ...
                "tx_index", tx_idx, ...
                "rx_index", rx_idx, ...
                "tx_name", string_value(get_prop(tx, "Name", sprintf("tx%d", tx_idx))), ...
                "rx_name", string_value(get_prop(rx, "Name", sprintf("rx%d", rx_idx))), ...
                "tx_position_m", vector3(get_prop(tx, "AntennaPosition", [])), ...
                "rx_position_m", vector3(get_prop(rx, "AntennaPosition", [])), ...
                "num_rays", numel(ray_list), ...
                "rays", {ray_records});
            link_index = link_index + 1;
        end
    end
end

function record = export_ray(ray, index)
    interactions = get_prop(ray, "Interactions", struct([]));
    record = struct();
    record.index = index;
    record.class = class(ray);
    record.path_specification = string_value(get_prop(ray, "PathSpecification", "Locations"));
    record.coordinate_system = string_value(get_prop(ray, "CoordinateSystem", "Cartesian"));
    record.system_scale = scalar_value(get_prop(ray, "SystemScale", 1), "ray.SystemScale");
    record.transmitter_location_m = vector3(get_prop(ray, "TransmitterLocation", []));
    record.receiver_location_m = vector3(get_prop(ray, "ReceiverLocation", []));
    record.line_of_sight = logical(get_prop(ray, "LineOfSight", false));
    record.frequency_hz = scalar_value(get_prop(ray, "Frequency", []), "ray.Frequency");
    record.path_loss_source = string_value(get_prop(ray, "PathLossSource", "Custom"));
    record.path_loss_db = scalar_value(get_prop(ray, "PathLoss", []), "ray.PathLoss");
    record.phase_shift_rad = scalar_value(get_prop(ray, "PhaseShift", []), "ray.PhaseShift");
    record.phase_shift_deg = rad2deg(record.phase_shift_rad);
    record.propagation_delay_s = scalar_value( ...
        get_prop(ray, "PropagationDelay", []), ...
        "ray.PropagationDelay");
    record.propagation_distance_m = scalar_value( ...
        get_prop(ray, "PropagationDistance", []), ...
        "ray.PropagationDistance");
    record.angle_of_departure_deg = angle_pair(get_prop(ray, "AngleOfDeparture", []));
    record.angle_of_arrival_deg = angle_pair(get_prop(ray, "AngleOfArrival", []));
    record.num_interactions = numel(interactions);
    record.interactions = export_interactions(interactions);
    record.path_coordinates_m = path_coordinates_or_sites( ...
        get_prop(ray, "PathCoordinates", []), ...
        record.transmitter_location_m, ...
        record.receiver_location_m, ...
        record.interactions);
end

function records = export_interactions(interactions)
    records = cell(1, numel(interactions));
    for idx = 1:numel(interactions)
        item = interactions(idx);
        records{idx} = struct( ...
            "index", idx, ...
            "class", class(item), ...
            "Type", string_value(get_prop(item, "Type", "")), ...
            "Location", vector3(get_prop(item, "Location", [NaN; NaN; NaN])), ...
            "MaterialName", string_value(get_prop(item, "MaterialName", "")));
    end
end

function record = export_propagation_model(propagation_model)
    record = struct( ...
        "class", class(propagation_model), ...
        "coordinate_system", string_value(get_prop( ...
            propagation_model, ...
            "CoordinateSystem", ...
            "cartesian")), ...
        "method", string_value(get_prop(propagation_model, "Method", "matlab_rt")), ...
        "max_num_reflections", scalar_or_default( ...
            get_prop(propagation_model, "MaxNumReflections", 0), ...
            0), ...
        "max_num_diffractions", scalar_or_default( ...
            get_prop(propagation_model, "MaxNumDiffractions", 0), ...
            0), ...
        "max_absolute_path_loss_db", optional_scalar( ...
            get_prop(propagation_model, "MaxAbsolutePathLoss", [])), ...
        "max_relative_path_loss_db", optional_scalar( ...
            get_prop(propagation_model, "MaxRelativePathLoss", [])));
end

function rays_cell = normalize_rays_cell(rays_output, num_tx, num_rx)
    if iscell(rays_output)
        rays_cell = rays_output;
    elseif num_tx == 1 && num_rx == 1
        rays_cell = {rays_output};
    else
        error("export_matlab_rt_json:ExpectedCellRays", ...
            "Multi-link raytrace output must be a cell array.");
    end

    if ~isequal(size(rays_cell), [num_tx, num_rx])
        if numel(rays_cell) == num_tx * num_rx
            rays_cell = reshape(rays_cell, [num_tx, num_rx]);
        else
            error("export_matlab_rt_json:RaysSizeMismatch", ...
                "Rays must have size num_tx-by-num_rx.");
        end
    end
end

function ray_list = normalize_ray_list(value)
    if isempty(value)
        ray_list = [];
    elseif iscell(value)
        ray_list = [value{:}];
    else
        ray_list = value(:).';
    end
end

function positions = site_positions_m(sites)
    positions = zeros(numel(sites), 3);
    for idx = 1:numel(sites)
        positions(idx, :) = vector3(get_prop(sites(idx), "AntennaPosition", []));
    end
end

function frequency_hz = site_frequency_hz(tx_sites)
    frequency_hz = scalar_value( ...
        get_prop(tx_sites(1), "TransmitterFrequency", []), ...
        "txsite.TransmitterFrequency");
end

function value = get_prop(object, name, default_value)
    try
        value = object.(name);
    catch
        value = default_value;
    end
end

function out = string_value(value)
    if isempty(value)
        out = "";
    else
        out = char(string(value));
    end
end

function out = scalar_value(value, name)
    if isempty(value)
        error("export_matlab_rt_json:MissingScalar", "%s is empty.", name);
    end
    out = double(value(1));
end

function out = scalar_or_default(value, default_value)
    if isempty(value)
        out = default_value;
    else
        out = double(value(1));
    end
end

function out = optional_scalar(value)
    if isempty(value)
        out = [];
    else
        out = double(value(1));
    end
end

function out = row_vector(value)
    out = double(value(:).');
end

function out = vector3(value)
    out = row_vector(value);
    if numel(out) ~= 3
        error("export_matlab_rt_json:InvalidVector3", "Expected a 3-element vector.");
    end
end

function out = angle_pair(value)
    out = row_vector(value);
    if numel(out) ~= 2
        error("export_matlab_rt_json:InvalidAnglePair", "Expected a 2-element angle vector.");
    end
end

function out = coordinate_rows(value)
    value = double(value);
    if isempty(value)
        error("export_matlab_rt_json:MissingPathCoordinates", "PathCoordinates is empty.");
    end
    if size(value, 1) == 3
        out = value.';
    elseif size(value, 2) == 3
        out = value;
    else
        error("export_matlab_rt_json:InvalidPathCoordinates", ...
            "PathCoordinates must be 3xN or Nx3.");
    end
end

function out = path_coordinates_or_sites(value, tx_location, rx_location, interactions)
    if ~isempty(value)
        out = coordinate_rows(value);
        return;
    end

    out = tx_location;
    for idx = 1:numel(interactions)
        out = [out; interactions{idx}.Location]; %#ok<AGROW>
    end
    out = [out; rx_location];
end

function merged = merge_structs(base, overrides)
    merged = base;
    if isempty(overrides)
        return;
    end

    names = fieldnames(overrides);
    for idx = 1:numel(names)
        merged.(names{idx}) = overrides.(names{idx});
    end
end

function text = encode_json(value)
    try
        text = jsonencode(value, PrettyPrint=true);
    catch
        text = jsonencode(value);
    end
end

function write_text(output_path, text)
    fid = fopen(output_path, "w");
    if fid < 0
        error("export_matlab_rt_json:OpenFailed", "Could not open JSON output path.");
    end
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, text, "char");
end

function ok = is_text_scalar(value)
    ok = ischar(value) || (isstring(value) && isscalar(value));
end

function ok = is_scalar_struct(value)
    ok = isstruct(value) && isscalar(value);
end
