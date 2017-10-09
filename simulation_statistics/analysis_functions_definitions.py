import numpy as np


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def radial_sample(in_matrix, samplenum):
    _, insize = in_matrix.shape
    centre = int(insize / 2. + .5 - 1)
    sampleradius = np.floor(insize / 2.)
    out = np.zeros(int(sampleradius))
    angles = np.linspace(0, 2 * np.pi, 100)
    dists = np.arange(0, sampleradius)
    for angle in angles:
        for dist in dists:
            tempx, tempy = pol2cart(angle, dist)
            yceil = int(np.ceil(tempy))
            yfloor = int(np.floor(tempy))
            xceil = int(np.ceil(tempx))
            xfloor = int(np.floor(tempx))
            if yceil == yfloor:
                if xceil == xfloor:
                    sample = in_matrix[
                        int(yceil + centre), int(xceil + centre)]
                else:
                    sample = in_matrix[yceil + centre, xfloor + centre] * \
                             np.mod(tempx, 1) + in_matrix[
                                                    yceil + centre, xceil + centre] * \
                                                (1 - np.mod(tempx, 1))
            else:
                if xceil == xfloor:
                    sample = in_matrix[yfloor + centre, xceil + centre] * \
                             np.mod(tempy, 1) + in_matrix[
                                                    yceil + centre, xceil + centre] * \
                                                (1 - np.mod(tempy, 1))
                else:
                    yfloorsample = in_matrix[
                                       yfloor + centre, xfloor + centre] * \
                                   np.mod(tempx, 1) + in_matrix[
                                                          yfloor + centre, xceil + centre] * \
                                                      (1 - np.mod(tempx, 1))
                    yceilsample = in_matrix[
                                      yceil + centre, xfloor + centre] * np.mod(
                        tempx, 1) + in_matrix[
                                        yceil + centre, xceil + centre] * (
                                        1 - np.mod(tempx, 1))
                    sample = yfloorsample * np.mod(tempy, 1) + yceilsample * (
                        1 - np.mod(tempy, 1))
            out[int(dist)] = out[int(dist)] + sample
    return out / float(samplenum)


# Function definitions
def conn_matrix_to_fan_in(conn_matrix, mode):
    conn_matrix = np.copy(conn_matrix)
    ys = int(np.sqrt(conn_matrix.shape[0]))
    xs = int(np.sqrt(conn_matrix.shape[1]))
    fan_in = np.zeros((ys ** 2, xs ** 2))
    locations = np.asarray(np.where(np.isfinite(conn_matrix)))

    for row in range(ys):
        for column in range(xs):
            if 'conn' in mode:
                fan_in[ys * row:ys * (row + 1),
                xs * column: xs * (column + 1)] = np.nan_to_num(
                    conn_matrix[:, row * xs + column].reshape(16, 16)) / g_max
            else:
                fan_in[ys * row:ys * (row + 1),
                xs * column: xs * (column + 1)] = np.nan_to_num(
                    conn_matrix[:, row * xs + column].reshape(16, 16))
    return fan_in


def centre_weights(in_star_all, n1d):
    in_star_all = np.copy(in_star_all)
    half_range = n1d // 2
    mean_projection = np.zeros((n1d + 1, n1d + 1))
    mean_centred_projection = np.zeros((n1d + 1, n1d + 1))
    positions = np.arange(-half_range, half_range + 1)
    means_and_std_devs = np.zeros((n1d ** 2, 8))
    means_for_plot = np.ones((n1d ** 2 * 2 - 1, 2)) * np.nan
    std_devs_xs = np.zeros(n1d)
    std_devs_ys = np.zeros(n1d)
    std_devs_xs_fine = np.zeros(11)
    std_devs_ys_fine = np.zeros(11)

    for y in range(n1d):
        for x in range(n1d):
            in_star = np.copy(
                in_star_all[y * n1d:(y + 1) * n1d, x * n1d:(x + 1) * n1d])
            in_star_extended = np.tile(in_star, [3, 3])
            if np.sum(in_star) > 0:
                # Add to the mean projection
                ideal_centred = np.copy(in_star_extended[
                                        n1d + y - half_range: n1d + y + half_range + 1,
                                        n1d + x - half_range:n1d + x + half_range + 1])
                ideal_centred[0, :] = ideal_centred[0, :] / 2.
                ideal_centred[n1d, :] = ideal_centred[n1d, :] / 2.
                ideal_centred[:, 0] = ideal_centred[:, 0] / 2.
                ideal_centred[:, n1d] = ideal_centred[:, n1d] / 2.

                mean_projection += ideal_centred

                #  ^^ So far so good ^^
                # Find the coarse centre of mass
                for pos in range(n1d):
                    temp_centred = np.copy(in_star_extended[
                                           n1d + pos - half_range: n1d + pos + half_range + 1,
                                           n1d + pos - half_range:n1d + pos + half_range + 1])
                    # correct the edges of centred
                    temp_centred[0, :] = temp_centred[0, :] / 2.
                    temp_centred[n1d, :] = temp_centred[n1d, :] / 2.
                    temp_centred[:, 0] = temp_centred[:, 0] / 2.
                    temp_centred[:, n1d] = temp_centred[:, n1d] / 2.
                    # calculate the StdDev
                    centred_x = np.sum(temp_centred, axis=0)
                    centred_y = np.sum(temp_centred, axis=1)
                    std_devs_xs[pos] = np.sqrt(
                        np.sum(centred_x * (positions ** 2)) / np.sum(
                            centred_x));
                    std_devs_ys[pos] = np.sqrt(
                        np.sum(centred_y * (positions ** 2)) / np.sum(
                            centred_y));

                std_dev_x = np.min(std_devs_xs)
                pos_x = np.argmin(std_devs_xs)
                std_dev_y = np.min(std_devs_ys)
                pos_y = np.argmin(std_devs_ys)

                #                 print pos_x, pos_y
                #                 print std_dev_x, std_dev_y



                # reconstruct the coarsely centred receptive field
                centred_coarse = np.copy(in_star_extended[
                                         n1d + pos_y - half_range:n1d + pos_y + half_range + 1,
                                         n1d + pos_x - half_range:n1d + pos_x + half_range + 1])
                centred_coarse[0, :] = centred_coarse[0, :] / 2.
                centred_coarse[n1d, :] = centred_coarse[n1d, :] / 2.
                centred_coarse[:, 0] = centred_coarse[:, 0] / 2.
                centred_coarse[:, n1d] = centred_coarse[:, n1d] / 2.

                for pos_fine in np.linspace(-.5, .5, 11):
                    assert std_devs_xs[
                               pos_x] == std_dev_x, "{0} != {1}".format(
                        std_devs_xs[pos_x], std_dev_x)
                    assert std_devs_ys[
                               pos_y] == std_dev_y, "{0} != {1}".format(
                        std_devs_ys[pos_y], std_dev_y)

                    temp_centred_fine = np.copy(in_star_extended[
                                                n1d + pos_y - half_range: n1d + pos_y + half_range + 1,
                                                n1d + pos_x - half_range:n1d + pos_x + half_range + 1])
                    # correct the edges of centred
                    temp_centred_fine[0, :] = temp_centred_fine[0, :] * (
                        .5 - pos_fine)
                    temp_centred_fine[n1d, :] = temp_centred_fine[n1d, :] * (
                        .5 + pos_fine)
                    temp_centred_fine[:, 0] = temp_centred_fine[:, 0] * (
                        .5 - pos_fine)
                    temp_centred_fine[:, n1d] = temp_centred_fine[:, n1d] * (
                        .5 + pos_fine)

                    # calculate the StdDev
                    centred_x = np.sum(temp_centred_fine, axis=0)
                    centred_y = np.sum(temp_centred_fine, axis=1)
                    positions_fine = np.arange(-half_range,
                                               half_range + 1) - pos_fine
                    positions_fine = positions_fine.flatten()
                    std_devs_xs_fine[
                        int(np.round(pos_fine * 10) + 5)] = np.sqrt(
                        np.sum(centred_x * (positions_fine ** 2)) / np.sum(
                            centred_x))
                    std_devs_ys_fine[
                        int(np.round(pos_fine * 10) + 5)] = np.sqrt(
                        np.sum(centred_y * (positions_fine ** 2)) / np.sum(
                            centred_y))

                # assert np.isclose(std_dev_x, std_devs_xs_fine[5]), "{0} != {1}".format(
                #     std_dev_x, std_devs_xs_fine[5])
                # assert np.isclose(std_dev_y, std_devs_ys_fine[5]), "{0} != {1}".format(
                #     std_dev_y, std_devs_ys_fine[5])
                std_dev_x = np.min(std_devs_xs_fine)
                pos_x_fine = np.argmin(std_devs_xs_fine)
                std_dev_y = np.min(std_devs_ys_fine)
                pos_y_fine = np.argmin(std_devs_ys_fine)
                pos_x_fine = (pos_x_fine - 5) / 10.
                pos_y_fine = (pos_y_fine - 5) / 10.

                # reconstruct the finely centred receptive field
                # and add to the mean centred projection
                second_to_first_indices = np.concatenate(
                    (np.arange(1, n1d + 1), [0]))
                last_to_first_indices = np.concatenate(
                    ([n1d], np.arange(0, n1d)))  # checked

                centred_left = centred_coarse[:, second_to_first_indices]
                centred_right = centred_coarse[:, last_to_first_indices]
                centred_fine_x = centred_left * np.max([0., -pos_x_fine]) + \
                                 centred_coarse * (1. - np.abs(pos_x_fine)) + \
                                 centred_right * np.max([0., pos_x_fine])

                centred_up = centred_fine_x[second_to_first_indices, :]
                centred_down = centred_fine_x[last_to_first_indices, :]
                centred_fine = centred_up * np.max([0., -pos_y_fine]) + \
                               centred_fine_x * (1. - np.abs(pos_y_fine)) + \
                               centred_down * np.max([0., pos_y_fine])

                mean_centred_projection += centred_fine

                std_dev = np.mean([std_dev_x, std_dev_y])
                mean_x = pos_x + pos_x_fine - x
                mean_y = pos_y + pos_y_fine - y
                if mean_x > half_range:
                    mean_x = mean_x - n1d
                if mean_x < -half_range:
                    mean_x = mean_x + n1d
                if mean_y > half_range:
                    mean_y = mean_y - n1d
                if mean_y < -half_range:
                    mean_y = mean_y + n1d
                mean_dist = np.sqrt(mean_x ** 2 + mean_y ** 2)
            else:
                mean_x = 0
                mean_y = 0
                mean_dist = 0
                std_dev = 0
            # For quiver plots
            if mean_dist == 0:
                means_and_std_devs[y * n1d + x, :] = np.asarray(
                    [x, y, mean_x, mean_y, mean_dist, std_dev, 0, 0])
            else:
                means_and_std_devs[y * n1d + x, :] = np.asarray(
                    [x, y, mean_x, mean_y, mean_dist, std_dev,
                     mean_x / mean_dist, mean_y / mean_dist])
                # For mapping plots
                Y = y + 1
                X = x + 1

                means_for_plot[(Y - 1) * n1d + X * np.remainder(Y, 2) +
                               (n1d + 1 - X) * np.remainder(Y - 1, 2) - 1,
                :] = [X + mean_x, Y + mean_y]
                means_for_plot[(X - 1) * n1d + Y * np.remainder(X - 1, 2) + (
                                                                            n1d + 1 - Y) * np.remainder(
                    X, 2) + n1d ** 2 - 1 - 1, :] = [X + mean_x, Y + mean_y]
                #     return (mean_projection/(n1d**2), std_dev)
    mean_projection = mean_projection / (n1d ** 2.)
    mean_centred_projection /= (n1d ** 2.)
    return (mean_projection, means_and_std_devs, means_for_plot,
            mean_centred_projection)


def fan_in(conn, weight, mode, area):
    conn = np.copy(conn).astype(np.int32)
    if 'rec' in area:
        conn[conn <= 255] = -1
    if 'ff' in area:
        conn[conn > 255] = -1
    output = np.zeros((256, 256))

    for syn in range(conn.shape[0]):
        for post_x in range(16):
            for post_y in range(16):
                pre_loc = int(conn[syn, post_x * 16 + post_y])
                if pre_loc >= 0:
                    pre_loc = np.mod(pre_loc, 256)
                    pre_x = int(np.floor(pre_loc / 16.))
                    pre_y = np.mod(pre_loc, 16)
                    #                     print pre_x, pre_y, post_x, post_y
                    #                     break
                    if 'conn' in mode:
                        output[post_x * 16 + pre_x, post_y * 16 + pre_y] += 1
                    else:
                        output[post_x * 16 + pre_x, post_y * 16 + pre_y] += \
                            weight[syn, post_x * 16 + post_y]
    return output


def distance(x0, x1, grid=np.asarray([16, 16]), type='euclidian'):
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    delta = np.abs(x0 - x1)
    delta = np.where(delta > grid * .5, delta - grid, delta)

    if type == 'manhattan':
        return np.abs(delta).sum(axis=-1)
    return np.sqrt((delta ** 2).sum(axis=-1))


def weight_shuffle(conn, weights, area):
    weights_copy = weights.copy()
    for post_id in range(weights_copy.shape[1]):
        pre_ids = conn[:, post_id]
        pre_weights = weights_copy[:, post_id]
        within_row_filter = np.argwhere(
            np.logical_and(pre_ids >= 0, pre_ids <= 255))
        permutation = np.random.permutation(within_row_filter)
        for index in range(within_row_filter.size):
            weights_copy[permutation[index], post_id] = weights[
                within_row_filter[index], post_id]

    return weights_copy


def list_to_post_pre(ff_list, lat_list, s_max, N_layer):
    conn = np.ones((s_max * 2, N_layer)) * -1
    weight = np.zeros((s_max * 2, N_layer))

    for target in range(N_layer):
        # source ids
        ff_pre_ids = ff_list[ff_list[:, 1] == target][:, 0]
        lat_pre_ids = lat_list[lat_list[:, 1] == target][:, 0] + N_layer
        conn[:ff_pre_ids.size + lat_pre_ids.size, target] \
            = np.concatenate((ff_pre_ids, lat_pre_ids))[:s_max * 2]
        # weights
        ff_pre_weights = ff_list[ff_list[:, 1] == target][:, 2]
        lat_pre_weights = lat_list[lat_list[:, 1] == target][:, 2]
        weight[:ff_pre_weights.size + lat_pre_weights.size, target] \
            = np.concatenate((ff_pre_weights, lat_pre_weights))[:s_max * 2]
    return conn, weight


def odc(fan_in_mat, mode=None):
    n1d = int(np.sqrt(fan_in_mat.shape[0]))
    odc_mask = np.zeros((n1d, n1d))
    for pre_y in range(n1d):
        for pre_x in range(n1d):
            odc_mask[pre_y, pre_x] = np.mod(pre_x + pre_y, 2)
    output = np.zeros((n1d, n1d))

    for post_y in range(n1d):
        for post_x in range(n1d):
            fan_in_temp = fan_in_mat[post_y * n1d:(post_y + 1) * n1d,
                        post_x * n1d:(post_x + 1) * n1d]
            if mode and 'NORMALISE' in mode.upper():
                temp = np.sum(np.sum(fan_in_temp * odc_mask)) / np.sum(
                    np.sum(np.logical(fan_in_temp * odc_mask))) / np.sum(
                    np.sum(fan_in_temp)) * np.sum(
                    np.sum(np.logical(fan_in_temp)))
                temp[np.where(np.isnan(temp))] = 1.
                output[post_y, post_x] = (1. / (1 + np.exp(-temp)) - 0.5) * 2
            else:
                output[post_y, post_x] = np.sum(
                    np.sum(fan_in_temp * odc_mask)) / np.sum(np.sum(fan_in_temp))

    output[np.where(np.isnan(output))] = .5
    return output