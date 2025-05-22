#!/bin/bash

# Script to set up test images for SMILEtrack tests

echo "Setting up test images for SMILEtrack tests..."

# Create image directory if it doesn't exist
mkdir -p image

# Check if test_image.jpg already exists
if [ -f "image/test_image.jpg" ]; then
    echo "Test image already exists at image/test_image.jpg"
else
    echo "Downloading test image..."
    # Download a Creative Commons image with multiple people
    # This is a sample URL - replace with an appropriate image URL
    curl -L "https://upload.wikimedia.org/wikipedia/commons/e/e0/People_watching_the_solar_eclipse_%2836761056432%29.jpg" -o image/test_image.jpg
    
    if [ $? -eq 0 ]; then
        echo "Test image downloaded successfully!"
    else
        echo "Error downloading test image. Please manually place an image with 4 people at image/test_image.jpg"
    fi
fi

echo "Setup complete! You can now run: cargo test test_detector_people_count" 