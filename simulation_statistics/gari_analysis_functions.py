from scipy import signal
from matplotlib.patches import Ellipse
import numpy as np


# Copied from connectivity_direction_orientation_stats.ipynb

def conns4post(conns, post_id):
    return conns[np.where(conns[:, 1] == post_id)[0], :]


def get_conns_for_posts(conns, posts_list):
    out = None
    first = True
    for post_id in posts_list:
        if first:
            out = conns4post(conns, post_id)
            first = False
        else:
            out = np.append(conns4post(conns, post_id))
    return out


def get_variance_ellipse(conns, width, height):
    '''Compute the variance of the location of the input neurons.
    :param conns: connections for A post-synaptic neuron, formated as FromListConnector
    :param width: width of the input population "image"
    :param height: height of the input population "image"
    :return centre: (x, y) the centre point of the variance ellipse
    :return shape: (width, height) the shape of the variance ellipse
    :return angle: orientation of the variance ellipse
    '''
    pres = conns[:, 0]
    ys, xs = pres // width, pres % width
    cov = np.cov(xs, ys)
    if len(np.where(np.isnan(cov))[0]) > 0 or \
            len(np.where(np.isinf(cov))[0]) > 0:
        return (0, 0), (0, 0), 0

    _lambda, v = np.linalg.eig(cov)
    _lambda = np.sqrt(_lambda)

    if len(np.where(np.isnan(_lambda))[0]) > 0 or \
            len(np.where(np.isinf(_lambda))[0]) > 0:
        return (0, 0), (0, 0), 0

    angle = np.rad2deg(np.arccos(v[0, 0]))
    centre_x, centre_y = np.mean(xs), np.mean(ys)
    ellipse_width, ellipse_height = _lambda[0], _lambda[1]

    return (centre_x, centre_y), (ellipse_width, ellipse_height), angle


def a2i(angle, angle_step):
    '''convert an angle (in degrees) into an index in the array
    :param angle: input angle
    :param angle_step: angle step with which the total angle (e.g. 360 degrees) r
                       ange is divided
    :return index: index in the array which corresponds to the input angle
    '''
    return np.int32(angle // angle_step)


def i2a(index, angle_step):
    '''convert an index in an array into an angle (in degrees)
    :param index: input index of the array
    :param angle_step: angle step with which the total angle (e.g. 360 degrees)
                       range is divided
    :return angle: angle which corresponds to the input index in the array
    '''

    return index * angle_step


def get_opp_ang(ang):
    '''get the opposite direction angle
    :param ang: preferred direction angle
    :return opp_ang: the opposite direction (pref+180 degrees) wrapped-around
                     360 degrees which keeps the angle in the range (0, 359)
                     when using integers
    '''

    return ((ang + 180) % 360)


def get_dists(pref_ang, angles, angle_step):
    '''get angular distances from the prefered direction at a sampling rate
    proportional to angle_step (wrapped around 360 degrees)
    :param pref_ang: preferred direction angle
    :param angles: angle list used in the experiments, should have been sampled
                   using angle_step
    :param angle_step: angle sampling step
    :return prf_d, opp_d: angular distances to the prefered direction and the
                          opposite direction, respectively
    '''
    opp_ang = get_opp_ang(pref_ang)

    prf_d = np.array(angles).copy()
    opp_d = np.array(angles).copy()

    prf_d -= 180
    prf_d[:] = np.abs(prf_d)
    prf_d[:] = np.roll(prf_d, opp_ang // angle_step)

    opp_d -= 180
    opp_d[:] = np.abs(opp_d)
    opp_d[:] = np.roll(opp_d, pref_ang // angle_step)

    return prf_d, opp_d


def exp_d(dist, sigma):
    '''helper to compute a normalized exponentially-decaying distance/gaussian
    :param dist: distance measurment
    :param sigma: standard deviation / how wide the gaussian should be
    :return exp distance:
    '''
    eee = np.exp(-(dist ** 2) / (2 * (sigma ** 2)))
    return eee / np.max(eee)


def get_model(per_ang_responses, pref_angle, angles, angle_step, sigma):
    '''dual exponential model for angle responses
    http://tips.vhlab.org/data-analysis/measures-of-orientation-and-direction-selectivity
    :param per_ang_responses: (average) response of neuron to a moving bar at
                              particular angle
    :param pref_angle: prefered direction
    :param angles: angle list used in the experiments
    :param angle_step: angle sampling step
    :param sigma: standard deviation / how wide the gaussian should be
    :return model:
    '''
    prf_d, opp_d = get_dists(pref_angle, angles, angle_step)
    prf_e, opp_e = exp_d(prf_d, sigma), exp_d(opp_d, sigma)
    prf_r = per_ang_responses[a2i(pref_angle, angle_step)]
    opp_r = per_ang_responses[a2i(get_opp_ang(pref_angle), angle_step)]

    return prf_r * prf_e + opp_r * opp_e


def build_pop_models(per_ang_responses, pop_size, pref_angle, angles,
                     angle_step, sigma):
    '''build double exponential model for each neuron in population
    http://tips.vhlab.org/data-analysis/measures-of-orientation-and-direction-selectivity
    :param per_ang_responses: (average) response of population to a moving bar at
                              particular angle
    :param pref_angle: prefered direction
    :param angles: angle list used in the experiments
    :param angle_step: angle sampling step
    :param sigma: standard deviation / how wide the gaussian should be
    :return models:
    '''
    models = np.empty((pop_size, angles.size))
    for neuron_id in range(pop_size):
        models[neuron_id, :] = get_model(per_ang_responses[neuron_id, :],
                                         pref_angle, angles, angle_step, sigma)
    return models


def conv_model(responses, width, sigma):
    '''build a circular convolution model using a Gaussian kernel (soften the
    signal)
    :param responses: (average) response of neuron to a moving bar at
                      particular angle
    :param width: kernel width
    :param sigma: Gaussian kernel standard deviation / wide
    '''
    hw = int(width // 2)
    tmp = np.empty(responses.size + 2 * hw)
    tmp[:hw] = responses[-hw:]
    tmp[-hw:] = responses[:hw]
    tmp[hw:-hw] = responses
    gsn = signal.gaussian(width, sigma)
    gsn /= gsn.sum()
    cnv = np.convolve(tmp, gsn, mode='same')
    return cnv[hw: -hw]


def get_wosi(model, pref_angle, angle_step, delta_angle=30):
    '''build a ranged Orientation Selectivity Index, instead of taking a single
    measurement at the prefered direction angle we take the sum of responses in
    the range of +/- delta_angle. This measures how much higher is the response
    to the preferred angle stimulus when compared to orthogonal angles
    (+/- 90 degrees).
    :param model: filtered responses per angle for A neuron
    :param pref_angle: prefered direction angle
    :param angle_step: angle sampling step for experiments
    :param delta_angle: angle range limit to sample responses (+/- delta deg)
    :return orientation selectivity index:
    '''
    prf_range = np.array([(pref_angle + a) % 360 \
                          for a in range(-delta_angle, delta_angle + 1, 1)])
    prfi = a2i(prf_range, angle_step)
    prfs = np.sum(model[prfi])

    ort_ang = (pref_angle + 90) % 360
    ort_range = np.array([(ort_ang + a) % 360 \
                          for a in range(-delta_angle, delta_angle + 1, 1)])

    orti = a2i(ort_range, angle_step)
    orts = np.sum(model[orti])

    ort_ang = (pref_angle - 90) % 360
    ortrange = np.array([(ort_ang + a) % 360 \
                         for a in range(-delta_angle, delta_angle + 1, 1)])
    orti = a2i(ort_range, angle_step)
    orts += np.sum(model[orti])

    osi = (prfs - orts) / prfs

    return np.clip(osi, 0, 1.0)


def get_osi(model, pref_angle, angle_step):
    '''compute Orientation Selectivity Index taking a single measurement at
    the prefered direction angle. This measures how much higher is the response
    to the preferred angle stimulus when compared to orthogonal angles
    (+/- 90 degrees).
    :param model: filtered responses per angle for A neuron
    :param pref_angle: prefered direction angle
    :param angle_step: angle sampling step for experiments
    :return orientation selectivity index:
    '''
    prfi = a2i(pref_angle, angle_step)

    ort_angle = (pref_angle + 90) % 360
    orti = a2i(ort_angle, angle_step)
    orts = model[orti]

    ort_angle = (pref_angle - 90) % 360
    orti = a2i(ort_angle, angle_step)
    orts += model[orti]

    osi = (model[prfi] - orts) / model[prfi]

    return np.clip(osi, 0., np.inf)


def get_wdsi(model, pref_angle, angle_step, delta_angle=30):
    '''build a ranged Direction Selectivity Index, instead of taking a single
    measurement at the prefered direction angle we take the sum of responses in
    the range of +/- delta_angle. This measures how much higher is the response
    to the preferred angle stimulus when compared to opposite (+180 degrees).
    :param model: filtered responses per angle for A neuron
    :param pref_angle: prefered direction angle
    :param angle_step: angle sampling step for experiments
    :param delta_angle: angle range limit to sample responses (+/- delta deg)
    :return direction selectivity index:
    '''

    prf_range = np.array([(pref_angle + a) % 360 \
                          for a in range(-delta_angle, delta_angle + 1, 1)])
    opp_ang = get_opp_ang(pref_angle)
    opp_range = np.array([(opp_ang + a) % 360 \
                          for a in range(-delta_angle, delta_angle + 1, 1)])

    prfi = a2i(prf_range, angle_step)
    oppi = a2i(opp_range, angle_step)

    prfs = np.sum(model[prfi])
    opps = np.sum(model[oppi])
    dsi = (prfs - opps) / prfs

    return np.clip(dsi, 0, np.inf)


def get_dsi(model, pref_angle, angle_step):
    '''compute Direction Selectivity Index taking a single measurement at
    the prefered direction angle. This measures how much higher is the response
    to the preferred angle stimulus when compared to opposite (+180 degrees).
    :param model: filtered responses per angle for A neuron
    :param pref_angle: prefered direction angle
    :param angle_step: angle sampling step for experiments
    :return direction selectivity index:
    '''
    opp_angle = (pref_angle + 180) % 360
    prfi = a2i(pref_angle, angle_step)
    oppi = a2i(opp_angle, angle_step)
    return (model[prfi] - model[oppi]) / model[prfi]


def build_all_dsi(responses, angles, angle_step, sigma,
                  do_double_gauss=False, wide=True, wide_delta_angle=30):
    model = np.empty_like(responses)
    dsi = np.zeros(responses.shape[0])
    for neuron_id in range(responses.shape[0]):
        pref_angle = i2a(np.argmax(responses[neuron_id]), angle_step)
        if do_double_gauss:
            model[neuron_id, :] = get_model(responses[neuron_id, :],
                                            pref_angle, angles, angle_step, sigma)
        else:
            model[neuron_id, :] = responses[neuron_id, :]

        if wide:
            dsi[neuron_id] = get_wdsi(model[neuron_id, :], pref_angle,
                                      angle_step, delta_angle=wide_delta_angle)
        else:
            dsi[neuron_id] = get_dsi(model[neuron_id, :], pref_angle, angle_step)

    return dsi


def get_random_posts(n_conns, width, height, roi=(8, 8, 24, 24)):
    # roi == region of interest (top, left, bottom, right)
    if roi is not None:
        rows = np.repeat(np.arange(roi[0], roi[2]), roi[3] - roi[1])
        cols = np.tile(np.arange(roi[1], roi[3]), roi[2] - roi[0])
    else:
        rows = np.arange(width)
        cols = np.arange(height)
    possible_ids = rows * width + cols
    return np.random.choice(possible_ids, size=n_conns, replace=False)


def get_local_max_idx(angle, angle_delta, responses, angle_step):
    '''Get the index at which the maximum response was found in a range
    around an angle.
    :param angle: central angle for region of interest
    :param angle_delta: range for region of interest (+/- delta)
    :param responses: response of a neuron to input at different angle
    :param angle_step: angle sampling step for experiments
    :return index:  at which maximum response was present:
    '''
    idx = a2i(angle, angle_step)
    idx_steps = angle_delta // angle_step
    indices = (np.arange(-idx_steps, idx_steps, 1) + idx) % len(responses)
    local_resp = responses[indices]
    max_idx = indices[np.argmax(local_resp)] % len(responses)
    return max_idx


def has_other_max(angle, angle_delta, max_rate, rate_delta, responses, angle_step):
    '''Find whether the response list has quantities higher than [max - delta]
    and they are outside the preferred response range (pref +/- delta).
    :param angle: preferred response angle
    :param angle_delta: limits for preferred response range
    :param max_rate: maximum rate expected (usually max of responses?)
    :param rate_delta: helps establish limit to consider other responses as
                       "maximum" ( i.e. > (max_rate - rate_delta) )
    :param responses: neuron response to input stimulus at different angles
    :param angle_step: angle sampling step for experiments
    :return bool(are there other max responses):
    '''
    idx = a2i(angle, angle_step)
    max_idx = a2i((angle + angle_delta) % 360, angle_step)
    min_idx = a2i((angle - angle_delta) % 360, angle_step)
    min_rate = max_rate - rate_delta
    if max_idx > min_idx:
        indices = np.append(np.arange(min_idx),
                            np.arange(max_idx, len(responses)))
    else:
        indices = np.arange(max_idx, min_idx)

    whr = np.where(responses[indices] > min_rate)[0]
    return len(whr) > 0

