#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:59:26 2024

@author: emanueleamato
"""


import os
import imageio
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize

"""

    Preprocesses images in the source folder and saves them as PNG files in the output folder.

    Args:

    - source_folder (str): Path to the folder containing the normalized images.
    - output_folder (str): Path to the folder where the preprocessed images will be saved.
    - height (int): Height of the resized images.
    - width (int): Width of the resized images.  

"""

class ImagePreprocessor:
    
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def resize_image(self, image):
        return resize(image, (self.height, self.width), preserve_range=True, mode='reflect', anti_aliasing=True)

    def data_norm(self, input):
        input = np.array(input, dtype=np.float32)
        input  = input - np.mean(input)
        output = input / (np.std(input) + 1e-12)
        return output

    def preprocess_and_save_images(self, source_folder, output_folder):
    
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                
                if file.lower().endswith('.nii'):
                    img_path = os.path.join(root, file)

                    img = sitk.ReadImage(img_path)
                    img_array = sitk.GetArrayFromImage(img)

                    for i in range(img_array.shape[0]):
                        slice_data = img_array[i, :, :]  # Estrai una singola slice
                        resized_slice = self.resize_image(slice_data)
                        normalized_slice = self.data_norm(resized_slice)
                        
                        # Normalizzo tra 0 ed 1 per questioni di salvataggio dati 
                        normalized_slice = (normalized_slice - np.min(normalized_slice)) / (np.max(normalized_slice) - np.min(normalized_slice))        
                        uint8_slice = (normalized_slice * 255).astype(np.uint8)
                    
                        # Salva la slice come PNG
                        output_filename = os.path.join(output_folder, f"preprocessed_img_{file}_{i}.png")
                        imageio.imwrite(output_filename, uint8_slice)
                        
     

# Esempio di utilizzo della classe
#preprocessor = ImagePreprocessor(height=512, width=512)
#preprocessor.preprocess_and_save_images(source_folder=source_folder, output_folder=source_folder)
