import os
import matplotlib.pyplot as plt
import time
from multiprocessing.dummy import Pool

import torch
import imageio

import math
from IPython.display import display


def get_timestamp(row):
    month, day, year = row["Date"].split("/")
    if int(month) < 10:
        month = "0" + month
    if int(day) < 10:
        day = "0" + day
    timestamp = f"{month}/{day}/{year} {row['Time']}"

    return timestamp


def row_to_imagepaths(row, img_dir):
    timestamp = get_timestamp(row)

    time_obj = time.strptime(timestamp, "%m/%d/%Y %H:%M:%S")
    date_dir = time.strftime("%Y-%m-%d", time_obj)
    time_dir = time.strftime("%H.%M.%S", time_obj)

    entries = os.scandir(f"{img_dir}\\{date_dir}\\HDR\\{time_dir}")
    jpgs = [
        entry.path
        for entry in entries
        if entry.is_file() and os.path.splitext(entry.name)[-1].lower() == ".jpg"
    ]
    jpgs.sort()

    if len(jpgs) < 8:
        raise BaseException("LESS THAN 8 PICTURES skip this sky")

    return jpgs[:8]


def get_samples(row, img_dir="D:\skies", radius=96):
    """
    Returns a (24, 100, 100) tensor
    The 8 samples stacked along the first dimension
    from the lowest exposure tensor[0:3] to the highest tensor[21:]
    """

    images = row_to_imagepaths(row, img_dir)
    image_arrs = torch.cat(
        [torch.from_numpy(imageio.imread(image)) for image in images], 2
    )

    height = int(image_arrs.shape[0])
    width = int(image_arrs.shape[1])
    channels = int(image_arrs.shape[2])

    crop_index = int((width - height) / 2)
    image_arrs = image_arrs[:, crop_index : crop_index + height, :]

    x = int(row["u"] * height)
    y = int(row["v"] * height)
    image_samples = image_arrs[y - radius : y + radius, x - radius : x + radius, :]

    for j in range(image_samples.shape[0]):
        for k in range(image_samples.shape[1]):
            y_offset = abs(j - radius)
            x_offset = abs(k - radius)
            if math.hypot(x_offset, y_offset) > radius:
                image_samples[j, k, :] = torch.zeros(channels)

    return image_samples.permute(2, 0, 1).float().contiguous() / 255


def get_samples_per_sky(df, img_dir="D:\skies", radius=50):
    first_row = df.iloc[0]
    skies = row_to_imagepaths(first_row, img_dir)
    skies = skies[3:4] + skies[5:]
    image_tensor = torch.cat(
        [torch.from_numpy(imageio.imread(image)) for image in skies], 2
    )

    height = int(image_tensor.shape[0])
    width = int(image_tensor.shape[1])
    channels = int(image_tensor.shape[2])

    crop_index = int((width - height) / 2)
    image_tensor = image_tensor[:, crop_index : crop_index + height, :]

    x = int(first_row["u"] * height)
    y = int(first_row["v"] * height)

    samples = image_tensor[
        y - radius : y + radius, x - radius : x + radius, :
    ].unsqueeze_(0)

    for _, row in df.iloc[1:].iterrows():
        x = int(row["u"] * height)
        y = int(row["v"] * height)

        samples = torch.cat(
            (
                samples,
                image_tensor[
                    y - radius : y + radius, x - radius : x + radius, :
                ].unsqueeze_(0),
            ),
            dim=0,
        )

    # for i in range(samples.shape[0]):  # batch
    for j in range(samples.shape[1]):  # y
        for k in range(samples.shape[2]):  # x
            y_offset = abs(j - radius)
            x_offset = abs(k - radius)
            if math.hypot(x_offset, y_offset) > radius:
                samples[:, j, k, :] = torch.zeros(channels)

    return samples.permute(0, 3, 1, 2).float().contiguous() / 255


def display_image(image_tensor):
    plt.imshow(image_tensor.permute(1, 2, 0).numpy())


def display_image_axle(ax, image_tensor):
    ax.imshow(image_tensor.permute(1, 2, 0).numpy())


def get_batch(df, index, batch_size, f):
    start = index * batch_size
    end = start + batch_size
    rows = df.iloc[start:end]

    with Pool(16) as p:
        batch = p.map(f, [rows.iloc[i] for i in range(batch_size)])

    return torch.stack(batch).contiguous()

