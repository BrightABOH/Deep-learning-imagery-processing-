#Import the neccesary modules

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from rasterio import mask

import os
import torch
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

from shapely.ops import unary_union

import rasterio
import geopandas as gpd
from shapely.geometry import shape
from sentinelhub import BBox, CRS, DataCollection, SentinelHubRequest, MimeType, SHConfig
from rasterio.plot import show
from torchvision import transforms
from PIL import Image
import numpy as np

import tarfile
import shutil

# Set your Sentinel Hub credentials
config = SHConfig()
config.sh_client_id = 'Add your Sentinel Hub instance ID'
config.sh_client_secret = 'Add your Sentinel Hub instance ID'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Set the path to your shapefile
shapefile_path = "/Users/brightabohsilasedem/Downloads/Agriculture_Sample_Data/Bugesera.shp"

# Set the date range for Sentinel-2 image search
start_date = date(2023, 7, 1)

end_date = date(2023, 9, 30)
#image size to be downloaded
image_size = (512, 512)
# Define the mean and standard deviation values for normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Set up Sentinel Hub API
bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "SCL", "B11", "B12","SCL"]
bands_s1 = ['VV', 'VH']
l_band = ["B02", "B03", "B04","B05"]
api = SentinelAPI(config.sh_client_id, config.sh_client_secret, 'https://scihub.copernicus.eu/dhus')
evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12","SCL"],
        output: [
            { id: "B01", bands: 1, sampleType: SampleType.AUTO },
            { id: "B02", bands: 1, sampleType: SampleType.AUTO },
            { id: "B03", bands: 1, sampleType: SampleType.AUTO },
            { id: "B04", bands: 1, sampleType: SampleType.AUTO },
            { id: "B05", bands: 1, sampleType: SampleType.AUTO },
            { id: "B06", bands: 1, sampleType: SampleType.AUTO },
            { id: "B07", bands: 1, sampleType: SampleType.AUTO },
            { id: "B08", bands: 1, sampleType: SampleType.AUTO },
            { id: "B8A", bands: 1, sampleType: SampleType.AUTO },
            { id: "B09", bands: 1, sampleType: SampleType.AUTO },
            { id: "B11", bands: 1, sampleType: SampleType.AUTO },
            { id: "B12", bands: 1, sampleType: SampleType.AUTO },
            { id: "RGB", bands: 3, sampleType: SampleType.AUTO },
            { id: "RGBN", bands: 4, sampleType: SampleType.AUTO },
            { id: "TCI", bands: 3, sampleType: SampleType.AUTO },
            { id: "NDVI", bands: 1, sampleType: SampleType.FLOAT32 },  // NDVI band
            { id: "SAVI", bands: 3, sampleType: SampleType.FLOAT32 },
            { id: "SCL", bands: 3, sampleType: SampleType.FLOAT32 },  
           
        ]
    };
}

function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    ndvi = (samples.B08 - samples.B04) / (samples.B08 + samples.B04);
    
    // Calculate SAVI
    L = 0.5;  // Soil brightness correction factor (adjust as needed)
    savi = ((samples.B08 - samples.B04) / (samples.B08 + samples.B04 + L)) * (1 + L);

    return {
        B01: [samples.B01],
        B02: [samples.B02],
        B03: [samples.B03],
        B04: [samples.B04],
        B05: [samples.B05],
        B06: [samples.B06],
        B07: [samples.B07],
        B08: [samples.B08],
        B8A: [samples.B8A],
        B09: [samples.B09],
        B11: [samples.B11],
        B12: [samples.B12],
        RGB: [2.5*samples.B04, 2.5*samples.B03, 2.5*samples.B02],
        RGBN: [samples.B04, samples.B03, samples.B02, samples.B08],
        TCI: [3*samples.B04, 3*samples.B03, 3*samples.B02],
        NDVI: [ndvi],  
        SAVI: [savi],
        SCL: [samples.SCL],
    };
}
"""
api = SentinelAPI(config.sh_client_id, config.sh_client_secret, 'https://scihub.copernicus.eu/dhus')
evalscript_s1 = """
//VERSION=3
function setup() {
    return {
        input: ["VV", "VH"],
        output: [
            { id: "VV", bands: 1, sampleType: SampleType.AUTO },
            { id: "VH", bands: 1, sampleType: SampleType.AUTO },
            { id: "RGB", bands: 3, sampleType: SampleType.AUTO }
            
        ],
        visualization: {
            bands: ["VV", "VH"],
            min: [-25,-25], // Adjust these values based on your data distribution
            max: [5,5], // Adjust these values based on your data distribution
        }
    };
}

function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    // Adjust the coefficients for a natural color representation
    ratio = samples.VH-samples.VV
    rgb_ratio = samples.VH+ samples.VV+ratio
    red = samples.VH;
    green = samples.VV;
    blue = rgb_ratio;
    return {
        VH: [red],
        VV: [green],
        RGB: [red, green, blue] 
    };
}
"""

api = SentinelAPI(config.sh_client_id, config.sh_client_secret, 'https://scihub.copernicus.eu/dhus')
evalscript_l8 = """
//VERSION=3
function setup() {
    return {
        input: ["B02", "B03", "B04","B05"], // Bands for true color and NIR
        output: [
            { id: "rgb", bands: 3,  sampleType: SampleType.AUTO}, // True color RGB
            { id: "ndvi", bands: 3,  sampleType: SampleType.AUTO} // NDVI
        ]
        
    };
}

function evaluatePixel(samples, scenes, inputMetadata, customData, outputMetadata) {
    // Calculate NDVI
        ndvi = (samples.B05 - samples.B04) / (samples.B05 + samples.B04);

    // Return true color RGB and NDVI values
    return {
        rgb: [2.5*samples.B04, 2.5*samples.B03, 2.5*samples.B02], // True color RGB
        ndvi: [ndvi] // NDVI
    };
}

"""
# Function to download Sentinel-2 images using sentinelhub
def download_sentinel_images(api, shapefile_path, start_date, end_date, output_folder):
    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.set_geometry('geometry')

    # Set the common CRS for both the shapefile and tiles
    common_crs = 'EPSG:32736'
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Calculate the area of each polygon in square meters
    target_crs = 'EPSG:32736'
    gdf = gdf.to_crs(target_crs)

    gdf['area_m2'] = gdf['geometry'].area

    # Calculate the bounding box of the union of all geometries in the shapefile
    shapefile_union = unary_union(gdf['geometry'])
    bbox = BBox(bbox=shape(shapefile_union).bounds, crs=CRS(common_crs))

    # Iterate over polygons
    for idx, row in gdf.iterrows():
        polygon = row['geometry']


        request = SentinelHubRequest(
            data_folder=os.path.join(output_folder, 'sentinel2'),
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date),
                    mosaicking_order='mostRecent',
                    maxcc=0.25
                )
            ],
            responses=[
                SentinelHubRequest.output_response('RGB', MimeType.TIFF),
                SentinelHubRequest.output_response('SCL', MimeType.TIFF)
            ],
            bbox=bbox,
            size=image_size,
            config=config
        )

        try:
            request.save_data()


            print(f"Data saved successfully for polygon {idx}!")
        except Exception as e:
            print(f"Error saving data for polygon {idx}: {e}")



def download_sentinel1_images(api, shapefile_path, start_date, end_date, output_folder):
    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.set_geometry('geometry')

    # Set the common CRS for both the shapefile and tiles
    common_crs = 'EPSG:32736'
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Calculate the area of each polygon in square meters
    target_crs = 'EPSG:32736'
    gdf = gdf.to_crs(target_crs)

    gdf['area_m2'] = gdf['geometry'].area
    # Check and print CRS of shapefile and raster data

    # Iterate over polygons
    for idx, row in gdf.iterrows():
        polygon = row['geometry']

        # Calculate the bounding box of the current polygon
        bbox = BBox(bbox=shape(polygon).bounds, crs=CRS(common_crs))

        request = SentinelHubRequest(
            data_folder=os.path.join(output_folder, 'sentinel1'),
            evalscript=evalscript_s1,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1_IW_DES,
                    time_interval=(start_date, end_date),
                    mosaicking_order='mostRecent',
                    maxcc=0.25
                )
            ],
            responses=[
                SentinelHubRequest.output_response('VV', MimeType.TIFF),
                SentinelHubRequest.output_response('VH', MimeType.TIFF),
                SentinelHubRequest.output_response('RGB', MimeType.TIFF),
            ],
            bbox=bbox,  # Use the bounding box of the current polygon
            size=image_size,
            config=config
        )

        try:
            request.save_data()
            print(f"Data saved successfully for polygon {idx}!")
        except Exception as e:
            print(f"Error saving data for polygon {idx}: {e}")

    

def download_landsat_images(api, shapefile_path, start_date, end_date, output_folder):
    # Load the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    gdf = gdf.set_geometry('geometry')


    # Set the common CRS for both the shapefile and tiles
    common_crs = 'EPSG:32736'
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Calculate the area of each polygon in square meters
    target_crs = 'EPSG:32736'
    gdf = gdf.to_crs(target_crs)

    gdf['area_m2'] = gdf['geometry'].area

    
    # Calculate the bounding box of the union of all geometries in the shapefile
    shapefile_union = unary_union(gdf['geometry'])
    bbox = BBox(bbox=shape(shapefile_union).bounds, crs=CRS(common_crs))

    # Iterate over polygons
    for idx, row in gdf.iterrows():
        polygon = row['geometry']


        request = SentinelHubRequest(
            data_folder=os.path.join(output_folder, 'landsat'),
            evalscript=evalscript_l8,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.LANDSAT_OT_L1,
                    time_interval=(start_date, end_date),
                    mosaicking_order='mostRecent',
                    maxcc=0.25

                )
            ],
            responses=[
                SentinelHubRequest.output_response('rgb', MimeType.TIFF),
                SentinelHubRequest.output_response('ndvi', MimeType.TIFF)
            ],
            bbox=bbox,
            size=image_size,
            config=config
        )

        try:
            request.save_data()


            print(f"Data saved successfully for polygon {idx}!")
        except Exception as e:
            print(f"Error saving data for polygon {idx}: {e}")
    
def extract_tar(tar_path, extract_path):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_path)

def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"File removed: {file_path}")
    except OSError as e:
        print(f"Error removing file {file_path}: {e}")

def preprocess_patch(patch):
    # Convert to RGB format
    patch_rgb = np.transpose(patch, (1, 2, 0))
    # Convert to PIL Image
    patch_image = Image.fromarray((patch_rgb * 255).astype(np.uint8))
    # Resize the image to match the model's input size
    patch_image = patch_image.resize((512, 512))
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # Convert to PyTorch tensor
    patch_tensor = transform(patch_image)
    # Add a batch dimension
    patch_tensor = patch_tensor.unsqueeze(0)
    return patch_tensor

def create_patches_from_single_image(image_path, patch_size=512):
    num_patches_total = 0
    total_predictions = np.zeros((1, 1, 512, 512))

    try:
        with rasterio.open(image_path) as src:
            num_patches = 0

            for i in range(0, src.width, patch_size):
                for j in range(0, src.height, patch_size):
                    window = rasterio.windows.Window(i, j, patch_size, patch_size)
                    patch = src.read(window=window)

                  
                    processed_patch = preprocess_patch(patch)

                    # Make predictions on the processed patch
                    with torch.no_grad():
                        predictions_patch = loaded_model(processed_patch.to(device))
                        predictions_patch = torch.sigmoid(predictions_patch)
                        predictions_patch = predictions_patch.cpu().numpy()

                        # Accumulate the predictions
                        total_predictions += predictions_patch

                    # For now, just print the patch information
                    print(f"Patch {num_patches + 1}: {window}")

                    num_patches += 1
                    num_patches_total += 1

            print(f"Number of patches created: {num_patches}")

    except rasterio.errors.RasterioIOError as e:
        print(f"Error opening the image at {image_path}: {e}")

# Function to visualize pixel distributions
def plot_pixel_distribution(data, title):
    plt.hist(data.flatten(), bins=range(0, 12), align='left', rwidth=0.8)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.show()

# Function to create colored RGB image with clouds and shadows replaced by Sentinel-1 VV band

def replace_clouds_and_shadow_with_sentinel1(rgb, s1_vv, cloud_mask):
    colored_rgb = rgb.copy()


     # Replace cloud pixels in RGB with corresponding values from Sentinel-1 VV
    colored_rgb[:, cloud_mask] = s1_vv[:, cloud_mask]
    colored_rgb[:, shadow_mask] = s1_vv[:, shadow_mask]
    # Replace shadow pixels in RGB with corresponding values from Sentinel-1 VV
    #colored_rgb[shadow_mask] = s1_vv[shadow_mask]

    return colored_rgb




def clip_and_save_image(image_path, shapefile_path, output_folder, output_filename):
    # Read the Sentinel-2 image
    with rasterio.open(image_path) as src:
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.set_geometry('geometry')

        # Convert the GeoDataFrame to the same CRS as the Sentinel-2 image
        gdf = gdf.to_crs(src.crs)

        # Use the bounds of the GeoDataFrame as the bounding box for clipping
        bbox = gdf.geometry.total_bounds

        # Perform the clipping
        clipped_image, transform = mask.mask(src, gdf.geometry, crop=True)

        # Update metadata for the clipped image
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                         "height": clipped_image.shape[1],
                         "width": clipped_image.shape[2],
                         "transform": transform})

        # Save the clipped image
        output_path = os.path.join(output_folder, output_filename)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(clipped_image)

    print(f"Clipped image saved to: {output_path}")



def open_landsat_image(output_folder, subfolder_name):
    # Get the subfolder path
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Traverse through subfolders using os.walk
    for root, dirs, files in os.walk(subfolder_path):
        # Check if 'rgb.tiff' is present in the current folder
        if 'rgb.tif' in files:
            rgb_file_path = os.path.join(root, 'rgb.tif')

            # Open the Sentinel-2 image
            with rasterio.open(rgb_file_path) as src:
                # Read the RGB bands
                l8_rgb = src.read([1, 2, 3], masked=True)
                show(l8_rgb)
                src.close
            return l8_rgb # Stop searching once the image is found

    print("RGB file not found in the specified subfolder.")

def open_sentinel2_image(output_folder, subfolder_name):
    # Get the subfolder path
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Traverse through subfolders using os.walk
    for root, dirs, files in os.walk(subfolder_path):
        # Check if 'rgb.tiff' is present in the current folder
        if 'RGB.tif' in files:
            s2_rgb_file_path = os.path.join(root, 'RGB.tif')

            # Open the Sentinel-2 image
            with rasterio.open(s2_rgb_file_path) as src:
                # Read the RGB bands
                rgb = src.read([1, 2, 3], masked=True)
                show(rgb)
                print('::::::')


            return rgb # Stop searching once the image is found

    print("RGB file not found in the specified subfolder.")


def open_scl_image(output_folder, subfolder_name):
    # Get the subfolder path
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Traverse through subfolders using os.walk
    for root, dirs, files in os.walk(subfolder_path):
        # Check if 'rgb.tiff' is present in the current folder
        if 'SCL.tif' in files:
            scl_file_path = os.path.join(root, 'SCL.tif')

            # Open the Sentinel-2 image
            with rasterio.open(scl_file_path) as src:

                # Read the RGB bands
                scl_band = src.read(1, masked=True)
                gdf = gpd.read_file(shapefile_path)
                gdf = gdf.set_geometry('geometry')

        # Convert the GeoDataFrame to the same CRS as the Sentinel-2 image
                gdf = gdf.to_crs(src.crs)
                scl_clip, transform = mask.mask(src, gdf.geometry, crop=True)

            # Update the metadata of the clipped image
                scl_meta = src.meta.copy()
                scl_meta.update({
                        "driver": "GTiff",
                        "height": scl_clip.shape[1],
                        "width": scl_clip.shape[2],
                        "transform": transform
                })


            return scl_band


            # Stop searching once the image is found

    print("RGB file not found in the specified subfolder.")

def open_scl_image_and_clip(output_folder, subfolder_name):
    # Get the subfolder path
    subfolder_path = os.path.join(output_folder, subfolder_name)

    # Traverse through subfolders using os.walk
    for root, dirs, files in os.walk(subfolder_path):
        # Check if 'rgb.tiff' is present in the current folder
        if 'SCL.tif' in files:
            scl_file_path = os.path.join(root, 'SCL.tif')

            # Open the Sentinel-2 image
            with rasterio.open(scl_file_path) as src:

                # Read the RGB bands
                scl_band = src.read(1, masked=True)
                gdf = gpd.read_file(shapefile_path)
                gdf = gdf.set_geometry('geometry')

        # Convert the GeoDataFrame to the same CRS as the Sentinel-2 image
                gdf = gdf.to_crs(src.crs)
                scl_clip, transform = mask.mask(src, gdf.geometry, crop=True)
                # Update the metadata of the clipped image
                scl_meta = src.meta.copy()
                scl_meta.update({
                        "driver": "GTiff",
                        "height": scl_clip.shape[1],
                        "width": scl_clip.shape[2],
                        "transform": transform
                })


def delete_subfolders(directory):
    # List all items (files and subdirectories) in the given directory
    items = os.listdir(directory)

    # Iterate over items
    for item in items:
        item_path = os.path.join(directory, item)

        # Check if the item is a subdirectory
        if os.path.isdir(item_path):
            try:
                # Use shutil.rmtree to delete the subdirectory and its contents
                shutil.rmtree(item_path)
                print(f"Subfolder '{item}' deleted successfully.")
            except Exception as e:
                print(f"Error deleting subfolder '{item}': {e}")

output_folder = "downloaded_image"
# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if __name__ == "__main__":

    img_folder = "img_folder"
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img_name = "constructed.tif"
    img_path = os.path.join(img_folder, img_name)


    # Step 1: Download Sentinel-2 images
    total_area_shapefile_hectares = download_sentinel_images(api, shapefile_path, start_date, end_date, output_folder)
    #Download landsat images
    download_landsat_images(api, shapefile_path, start_date, end_date, output_folder)
    #Step 2: Download Sentinel-1 images
    #download_sentinel1_images(api, shapefile_path, start_date, end_date, output_folder)

    for root, dirs, files in os.walk(output_folder):

        for file in files:
            if file.endswith(".tar"):
                tar_path = os.path.join(root, file)
                extract_path = os.path.join(root, file[:-4])  # Remove '.tar' extension
                extract_tar(tar_path, extract_path)
                remove_file(tar_path)




    open_landsat_image(output_folder, 'landsat')
    open_sentinel2_image(output_folder,'sentinel2')
    open_scl_image(output_folder,'sentinel2')

    #Define the crop_land prediction threshold
    #cropland_threshold = 0.6
    # Define thresholds for cloud and shadow pixels in SCL band
    cloud_threshold = [8,9]
    shadow_threshold = [3]


    cloud_mask = np.isin(open_scl_image(output_folder,'sentinel2'), cloud_threshold)
    shadow_mask = np.isin(open_scl_image(output_folder,'sentinel2'), shadow_threshold)


    rgb = open_sentinel2_image(output_folder,'sentinel2')
    l8_rgb = open_landsat_image(output_folder,'landsat')



            # Visualize pixel distribution of the cloud mask
    plot_pixel_distribution(cloud_mask, 'Cloud Mask Distribution')
    plot_pixel_distribution(shadow_mask, 'Shadow Mask Distribution')

    rgb_replaced = replace_clouds_and_shadow_with_sentinel1(rgb,l8_rgb,cloud_mask)


   

    # Visualize the original and modified RGB images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(np.moveaxis(rgb.data, 0, -1))
    axes[0].set_title('Original RGB Image with cloud covers')
    axes[1].imshow(np.moveaxis(rgb_replaced.data, 0, -1))
    axes[1].set_title('RGB Image with Clouds Replaced by Landsat 8 pixels')
    plt.show()

    # Copy georeferencing information from the original Sentinel-2 image
    print("Stopping here!")
    # Print the paths for debugging


    # Update the s2_rgb_file_path to the correct path
    s2_rgb_file_path = os.path.join(root,'response','RGB.tif')
    print(f"s2_rgb_file_path: {s2_rgb_file_path}")
    print(f"img_path: {img_path}")
    # Open the source file
    with rasterio.open(s2_rgb_file_path) as src:
        transform = src.transform
        crs = src.crs

        # Now, open the destination file
        try:
            with rasterio.open(img_path, 'w', driver='GTiff', height=src.height, width=src.width, count=3, dtype=rgb_replaced.dtype, crs=crs, transform=transform) as dst:
                # Write the replaced data to the destination file
                dst.write(rgb_replaced)
        except Exception as e:
            print(f"Error opening destination file: {e}")

    print(f"Reconstructed image saved at: {img_path}")





        # Save the clipped image with the name of the shapefile
    clip_and_save_image(img_path,shapefile_path, output_folder, f"{os.path.splitext(os.path.basename(shapefile_path))[0]}_clipped.tiff")
    scl_band = open_scl_image(output_folder,'sentinel2')
    
    downloaded_image_path = os.path.join(output_folder, os.listdir(output_folder)[0])
    












