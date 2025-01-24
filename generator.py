import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import cv2
import os
from PIL import Image

# Load the star data
file_path = 'hip.txt'  # Replace with the actual file path
df = pd.read_csv(file_path, delimiter='|', skipinitialspace=True)
df = df.drop(df.columns[[0, -1]], axis=1)
df.columns = ['name', 'spect_type', 'vmag', 'ra_deg', 'dec_deg', 'bv_color']

# Filter stars for the Northern Hemisphere (Declination >= 0)
df_north = df[df['dec_deg'] >= 0]

# Load constellation boundary data
url_bounds = "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.bounds.json"
response_bounds = requests.get(url_bounds)
constellation_bounds = response_bounds.json()

# Load constellation lines data
url_lines = "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.lines.json"
response_lines = requests.get(url_lines)
constellation_lines = response_lines.json()

# Create a directory to save cropped images
output_dir = 'images'
os.makedirs(output_dir, exist_ok=True)


def shortest_path_interpolation(start_ra, end_ra, num_points):
    # Calculate the difference between the two angles
    delta_ra = (end_ra - start_ra) % (2 * np.pi)

    # Adjust for the shortest path
    if delta_ra > np.pi:
        delta_ra -= 2 * np.pi

    # Generate the interpolated points
    ra_values = np.linspace(0, delta_ra, num_points) + start_ra
    ra_values = ra_values % (2 * np.pi)  # Ensure RA values are within [0, 360)

    return ra_values


def interpolate(ra_polygon, dec_polygon):
    interpolated_ra = [ra_polygon[0], ]
    interpolated_dec = [dec_polygon[0], ]

    for i in range(1, len(ra_polygon)):
        if abs(ra_polygon[i] - ra_polygon[i - 1]) > 0:
            for ra in shortest_path_interpolation(ra_polygon[i - 1], ra_polygon[i],
                                                  int(abs(ra_polygon[i] - ra_polygon[i - 1]) / np.pi * 180 * 8)):
                interpolated_ra.append(ra)
                interpolated_dec.append(dec_polygon[i])
        interpolated_ra.append(ra_polygon[i])
        interpolated_dec.append(dec_polygon[i])

    return np.array(interpolated_ra), np.array(interpolated_dec)

def interpolate_line(ra_polygon, dec_polygon):
    interpolated_ra = [ra_polygon[0], ]
    interpolated_dec = [dec_polygon[0], ]

    for i in range(1, len(ra_polygon)):
        for percent in np.linspace(0, 1, 10):
            interpolated_ra.append(ra_polygon[i - 1] * (1-percent) + ra_polygon[i] * percent)
            interpolated_dec.append(dec_polygon[i - 1] * (1-percent) + dec_polygon[i] * percent)

    return np.array(interpolated_ra), np.array(interpolated_dec)

def clip_polygon_to_radial_distance(ra_polygon, dec_polygon, radial_distance_limit=90):
    clipped_ra = []
    clipped_dec = []

    previous = [9999, 9999]
    outside_limits = False

    for i in range(len(ra_polygon)):
        if not outside_limits:
            if dec_polygon[i] > radial_distance_limit:
                clipped_ra.append(ra_polygon[i])
                clipped_dec.append(89.6)
                previous = [ra_polygon[i], 89.6]
                outside_limits = True
                continue
            previous = [ra_polygon[i], dec_polygon[i]]
            clipped_ra.append(ra_polygon[i])
            clipped_dec.append(dec_polygon[i])
        elif outside_limits and dec_polygon[i] < radial_distance_limit:
            outside_limits = False
            if previous[0] == 9999:
                continue
            for ra in shortest_path_interpolation(previous[0], ra_polygon[i], 50):
                clipped_ra.append(ra)
                clipped_dec.append(89.6)
            clipped_ra.append(ra_polygon[i])
            clipped_dec.append(dec_polygon[i])

            previous = [ra_polygon[i], dec_polygon[i]]
    if previous[0] != 9999 and outside_limits:
        for ra in shortest_path_interpolation(previous[0], ra_polygon[0], 50):
            clipped_ra.append(ra)
            clipped_dec.append(89.6)
    return np.array(clipped_ra), np.array(clipped_dec)


# Create the full star map with a polar projection
fig_full, ax_full = plt.subplots(figsize=(20, 20), subplot_kw={'projection': 'polar'})

for feature, bounds in zip(sorted(constellation_lines['features'], key=lambda x: x["id"] != "Ser"),
                           sorted(constellation_bounds['features'], key=lambda x: x["id"] != "Ser")):
    coordinates = bounds["geometry"]["coordinates"][0]

    ra_polygon = np.radians([point[0] for point in coordinates])  # RA to radians
    dec_polygon = 90 - np.array([point[1] for point in coordinates])  # Declination to radial distance

    ra_polygon, dec_polygon = interpolate(ra_polygon, dec_polygon)

    ra_polygonc, dec_polygonc = clip_polygon_to_radial_distance(ra_polygon, dec_polygon)

    # Create a Path from the polygon coordinates
    polygon_pathc = Path(np.column_stack([ra_polygonc, dec_polygonc]))

    # Create a PathPatch from the polygon
    patch = PathPatch(polygon_pathc, facecolor='none', edgecolor='yellow', linewidth=0.2)
    ax_full.add_patch(patch)  # Add the patch to the polar axes


ax_full.set_theta_zero_location('N')  # Set 0 degrees (North) at the top
ax_full.set_theta_direction(-1)  # Reverse the direction of RA (clockwise)
ax_full.set_rlim(0, 90)  # Radial limits (90° at the center, 0° at the edge)
ax_full.axes.yaxis.set_ticks([])
ax_full.axes.xaxis.set_ticks([])

# Save the plot as an image
temp_image_path = os.path.join(output_dir, f'map.png')
plt.savefig(temp_image_path, bbox_inches='tight', pad_inches=0,
            transparent=True)  # Save with transparent background
plt.close(fig_full)


result_labels = []

for ind, (feature, bounds) in enumerate(zip(sorted(constellation_lines['features'], key=lambda x: x["id"]),
                                            sorted(constellation_bounds['features'], key=lambda x: x["id"]))):
    # Create the full star map with a polar projection
    fig_full, ax_full = plt.subplots(figsize=(20, 20), subplot_kw={'projection': 'polar'})

    # Convert RA from degrees to radians (required for polar plots)
    ra_rad = np.radians(df_north['ra_deg'])

    # Convert Declination to radial distance (90° at the center, 0° at the edge)
    radial_distance = 90 - df_north['dec_deg']

    # Plot the stars on the full map
    scatter = ax_full.scatter(ra_rad, radial_distance, s=10 ** (-0.4 * df_north["vmag"]) * 10, c="black", alpha=0.8)

    # Plot constellation borders on the full map
    lines = feature['geometry']['coordinates']
    for line in lines:
        ra_line = np.radians([point[0] for point in line])
        dec_line = 90 - np.array([point[1] for point in line])
        ax_full.plot(ra_line, dec_line, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Extract polygon coordinates for the constellation boundary
    coordinates = bounds["geometry"]["coordinates"][0]

    # Convert polygon coordinates to polar coordinates
    ra_polygon = np.radians([point[0] for point in coordinates])  # RA to radians
    dec_polygon = 90 - np.array([point[1] for point in coordinates])  # Declination to radial distance

    ra_polygon, dec_polygon = interpolate(ra_polygon, dec_polygon)

    # Clip the polygon to the radial distance limit
    ra_polygon, dec_polygon = clip_polygon_to_radial_distance(ra_polygon, dec_polygon)

    # Create a Path from the polygon coordinates
    polygon_path = Path(np.column_stack([ra_polygon, dec_polygon]))

    # Create a PathPatch from the polygon
    patch = PathPatch(polygon_path, facecolor='none', edgecolor='red', linewidth=0.7)
    ax_full.add_patch(patch)  # Add the patch to the polar axes

    # Clip the scatter plot to the polygon
    for collection in ax_full.collections:
        collection.set_clip_path(patch)

    # Customize the full plot
    ax_full.set_theta_zero_location('N')  # Set 0 degrees (North) at the top
    ax_full.set_theta_direction(-1)  # Reverse the direction of RA (clockwise)
    ax_full.set_rlim(0, 90)  # Radial limits (90° at the center, 0° at the edge)
    # ax_full.axes.yaxis.set_ticks([])
    # ax_full.axes.xaxis.set_ticks([])
    ax_full.axis('off')

    # Save the plot as an image
    temp_image_path = os.path.join(output_dir, f'{bounds["id"]}{ind}_temp_plot.png')
    plt.savefig(temp_image_path, bbox_inches='tight', pad_inches=0,
                transparent=True)  # Save with transparent background
    plt.close(fig_full)

    # Load the saved image with OpenCV
    image = cv2.imread(temp_image_path, cv2.IMREAD_UNCHANGED)

    image_height = image.shape[0]

    # Convert the image to HSV color space for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])  # Red wraps around in HSV
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        # Find the largest contour (assuming it's the closed red polyline)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polyline
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx_polyline = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Check if the polyline is closed (has more than 2 points)
        if len(approx_polyline) > 2:
            # Get the bounding box of the polyline
            x, y, w, h = cv2.boundingRect(approx_polyline)

            # Crop the image using the bounding box
            cropped_image = image[y:y + h, x:x + w]

            # Create a mask for the polygon
            mask = np.zeros_like(image[:, :, 0])  # Create a single-channel mask
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Apply the mask to the cropped image
            cropped_mask = mask[y:y + h, x:x + w]
            cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

            # Convert the cropped image to RGBA (to add transparency)
            cropped_image_rgba = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGBA)

            # Set the alpha channel to 0 for areas outside the mask
            cropped_image_rgba[:, :, 3] = cropped_mask

            # Save the cropped image with transparency
            constellation_name = bounds["id"]
            cropped_image_path = os.path.join(output_dir, f'{constellation_name}{ind}_cropped.png')
            cv2.imwrite(cropped_image_path, cropped_image_rgba)

            result_labels.append({"name": f'{constellation_name}{ind}_cropped.png', "id":constellation_name, "x": x / image_height, "y": y / image_height, "w": w / image_height, "h": h / image_height})

            print(f"Cropped image saved as '{cropped_image_path}'")
        else:
            print(f"No closed red polyline found for constellation {bounds['id']}{ind}.")
    else:
        print(f"No red contours found for constellation {bounds['id']}{ind}.")

    # Remove the temporary image
    os.remove(temp_image_path)
with open(os.path.join(output_dir, f'labels.json'), "w") as f:
    json.dump(result_labels, f)