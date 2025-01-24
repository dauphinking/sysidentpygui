import piexif
from PIL import Image

# Function to remove specific metadata
def remove_metadata(image_path, output_path):
    # Open the image
    image = Image.open(image_path)
    
    # Check if EXIF data is present
    if 'exif' in image.info:
        exif_dict = piexif.load(image.info['exif'])
        # Print the exif data for inspection
        print(exif_dict)
    else:
        print("No EXIF data found.")
    
    # Remove owner and computer information
    # Commented out for now
    # Convert back to bytes
    # exif_bytes = piexif.dump(exif_dict)
    
    # Save the image without the metadata
    # image.save(output_path, "jpeg", exif=exif_bytes)

# Example usage
remove_metadata('input.jpg', 'output.jpg') 