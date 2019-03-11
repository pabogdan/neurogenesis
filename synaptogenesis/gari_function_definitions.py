import numpy as np
import os
import bz2

def load_compressed_spikes(name):
    fname = os.path.join(os.getcwd(), '%s.bz2' % name)
    print(fname)
    spikes = []
    with bz2.BZ2File(fname, 'rb') as f:
        for line in f:
            spl = line.split(' ')
            spikes.append((int(float(spl[0])), float(spl[1])))

        return spikes


def rowcol(indices, row_bits=5, col_bits=5, chann_bits=1):
    '''from X | Y | P'''
    row = np.bitwise_and(
        np.right_shift(indices, chann_bits), ((1 << row_bits) - 1))
    col = np.bitwise_and(
        np.right_shift(indices, (row_bits + chann_bits)),
        ((1 << col_bits) - 1))
    print(np.unique(col))
    chn = np.bitwise_and(indices, ((1 << chann_bits) - 1))
    return row, col, chn


def split_in_spikes(spikes, row_bits=5, col_bits=5, chann_bits=1, width=32):
    rows, cols, chans = rowcol(np.array([np.uint32(i) for i, _ in spikes]),
                               row_bits=row_bits, col_bits=col_bits,
                               chann_bits=chann_bits)
    #     print(rows, cols, chans)
    full_times = np.array([t for _, t in spikes])
    on_indices = np.where(chans == 1)[0]
    on_ids = rows[on_indices] * width + cols[on_indices]
    on_ts = full_times[on_indices]

    off_indices = np.where(chans == 0)[0]
    off_ids = rows[off_indices] * width + cols[off_indices]
    off_ts = full_times[off_indices]

    on_spks = sorted([[on_ids[i], on_ts[i]] for i in range(len(on_ts))],
                     key=lambda x: x[1])
    off_spks = sorted([[off_ids[i], off_ts[i]] for i in range(len(off_ts))],
                      key=lambda x: x[1])

    return on_spks, off_spks


def img_map(idx, width, n_per_coord=1):
    row = idx // (width * n_per_coord)
    col = (idx - row * width * n_per_coord) // n_per_coord
    return row, col


def spikes_to_images(on_spikes, off_spikes, width, height, n_per_coord=1,
                     dt=11, cols=10):
    max_t = np.max([t for _, t in on_spikes])
    print "on_spikes length = ", len(on_spikes)
    if off_spikes is not None:
        max_t = max(max_t, np.max([t for _, t in off_spikes]))
        print "off_spikes length = ", len(off_spikes)
    max_t = int(max_t)

    img = np.zeros((height, width, 3))
    imgs = []
    start_t = 0
    on_start = 0
    off_start = 0
    for start_t in range(0, max_t, dt):
        end_t = start_t + dt
        for i, spike in enumerate(on_spikes[on_start:]):
            idx, t = spike
            if start_t <= t and t < end_t:
                row, col = img_map(idx, width, n_per_coord)
                img[row, col, 1] += 10.
            else:
                on_start += i
                break

        if off_spikes is not None:
            for j, spike in enumerate(off_spikes[off_start:]):
                idx, t = spike
                if start_t <= t and t < end_t:
                    row, col = img_map(idx, width, n_per_coord)
                    img[row, col, 0] += 10.
                else:
                    off_start += j
                    break

        imgs.append(img.copy())
        img[:] = 0.

    return imgs


def xyp2ssa(spike_list, width, height, n_channels=1):
    ID, t = 0, 1
    row_bits = int(np.ceil(np.log2(height)))
    col_bits = int(np.ceil(np.log2(width)))
    chann_bits = int(np.ceil(np.log2(n_channels)))
    row_mask = (1 << row_bits) - 1
    col_mask = (1 << col_bits) - 1
    chan_mask = (1 << chann_bits) - 1
    new_spike_list = [[[] for _ in range(width * height)]
                      for _ in range(n_channels)]
    for spike in spike_list:
        row = (spike[ID] >> chann_bits) & row_mask
        col = (spike[ID] >> (chann_bits + row_bits)) & col_mask
        ch = spike[ID] & chan_mask
        nid = row * width + col
        new_spike_list[ch][nid].append(spike[t])

    for ch in range(len(new_spike_list)):
        for nid in range(len(new_spike_list[ch])):
            new_spike_list[ch][nid].sort()

    return new_spike_list[0]
