#!/usr/bin/env python3
"""
Digital Image Processing Suite
==============================

A comprehensive image processing toolkit that includes:
- Image encryption/decryption using XOR operations
- Wiener filtering for noise reduction
- Histogram analysis (grayscale and RGB)
- Interactive image processing pipeline

Author: Your Name
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.signal import wiener
import imageio
import os
import sys
from typing import Optional, Tuple


class ImageProcessor:
    """Main class for image processing operations."""
    
    def __init__(self, default_image_path: str = "image.png"):
        """
        Initialize the ImageProcessor.
        
        Args:
            default_image_path: Default path to use when no image is specified
        """
        self.default_image_path = default_image_path
        self.current_image = None
        self.encryption_key = None
        
    def load_image(self, path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Load an image with error handling and ensure RGB mode.
        
        Args:
            path: Path to the image file. If None, prompts user for input.
            
        Returns:
            numpy array of the image or None if loading failed
        """
        if path is None:
            path = input("Enter the image path (press Enter for default): ").strip()
            
        if not path:
            path = self.default_image_path
            
        if not os.path.exists(path):
            print(f"Error: File '{path}' not found!")
            return None
            
        try:
            image = np.array(Image.open(path).convert("RGB"))
            self.current_image = image
            print(f"Successfully loaded image: {path}")
            print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} pixels")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def display_image(self, image: np.ndarray, title: str = "Image", save_path: Optional[str] = None):
        """
        Display an image using matplotlib.
        
        Args:
            image: Image array to display
            title: Title for the plot
            save_path: Optional path to save the displayed image
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(title, fontsize=16)
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Image saved to: {save_path}")
            
        plt.show()
    
    def generate_encryption_key(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate a random encryption key for XOR operations.
        
        Args:
            shape: Shape of the image (height, width, channels)
            
        Returns:
            Random key array
        """
        np.random.seed(42)  # For reproducible results
        key = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        self.encryption_key = key
        return key
    
    def encrypt_image(self, image: np.ndarray, save_path: str = "encrypted_image.png") -> np.ndarray:
        """
        Encrypt an image using XOR operation.
        
        Args:
            image: Input image array
            save_path: Path to save the encrypted image
            
        Returns:
            Encrypted image array
        """
        image = image.astype(np.uint8)
        key = self.generate_encryption_key(image.shape)
        
        encrypted = np.bitwise_xor(image, key)
        
        # Save encrypted image
        encrypted_img = Image.fromarray(encrypted)
        encrypted_img.save(save_path)
        print(f"Encrypted image saved to: {save_path}")
        
        return encrypted
    
    def decrypt_image(self, encrypted_image: np.ndarray, save_path: str = "decrypted_image.png") -> np.ndarray:
        """
        Decrypt an image using XOR operation.
        
        Args:
            encrypted_image: Encrypted image array
            save_path: Path to save the decrypted image
            
        Returns:
            Decrypted image array
        """
        if self.encryption_key is None:
            raise ValueError("No encryption key available. Encrypt an image first.")
        
        decrypted = np.bitwise_xor(encrypted_image, self.encryption_key)
        
        # Save decrypted image
        decrypted_img = Image.fromarray(decrypted)
        decrypted_img.save(save_path)
        print(f"Decrypted image saved to: {save_path}")
        
        return decrypted
    
    def apply_wiener_filter(self, image_path: Optional[str] = None, save_path: str = "wiener_filtered.png") -> Optional[np.ndarray]:
        """
        Apply Wiener filter for noise reduction.
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the filtered image
            
        Returns:
            Filtered image array or None if failed
        """
        if image_path is None:
            image_path = input("Enter path for Wiener filtering (press Enter for default): ").strip()
            
        if not image_path:
            image_path = self.default_image_path
            
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found!")
            return None
        
        try:
            # Load image using imageio
            image = imageio.imread(image_path)
            
            # Apply Wiener filter
            filtered_image = wiener(image)
            
            # Save filtered image
            imageio.imwrite(save_path, filtered_image.astype(np.uint8))
            print(f"Wiener filtered image saved to: {save_path}")
            
            return filtered_image
        except Exception as e:
            print(f"Error applying Wiener filter: {e}")
            return None
    
    def plot_grayscale_histogram(self, image_path: Optional[str] = None, save_path: str = "grayscale_histogram.png"):
        """
        Plot histogram of a grayscale image.
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the histogram plot
        """
        if image_path is None:
            image_path = input("Enter path for grayscale histogram (press Enter for default): ").strip()
            
        if not image_path:
            image_path = self.default_image_path
            
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found!")
            return
        
        try:
            # Load image as grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Error: Could not read the image in grayscale!")
                return
            
            # Calculate histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            
            # Plot histogram
            plt.figure(figsize=(12, 6))
            plt.title("Grayscale Histogram", fontsize=16)
            plt.xlabel("Gray Level", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.bar(range(256), hist.ravel(), color="black", alpha=0.7)
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Grayscale histogram saved to: {save_path}")
            plt.show()
            
        except Exception as e:
            print(f"Error generating grayscale histogram: {e}")
    
    def plot_rgb_histograms(self, image_path: Optional[str] = None, save_path: str = "rgb_histograms.png"):
        """
        Plot RGB component histograms.
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the histogram plots
        """
        if image_path is None:
            image_path = input("Enter path for RGB histogram (press Enter for default): ").strip()
            
        if not image_path:
            image_path = self.default_image_path
            
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found!")
            return
        
        try:
            # Open image and convert to RGB
            img = Image.open(image_path).convert("RGB")
            r, g, b = img.split()
            
            plt.figure(figsize=(15, 5))
            colors = ["red", "green", "blue"]
            channels = [r, g, b]
            titles = ["Red Component Histogram", "Green Component Histogram", "Blue Component Histogram"]
            
            for i, (channel, color, title) in enumerate(zip(channels, colors, titles)):
                plt.subplot(1, 3, i + 1)
                plt.hist(np.array(channel).ravel(), bins=256, color=color, alpha=0.6, edgecolor='black')
                plt.title(title, fontsize=14)
                plt.xlabel("Pixel Value", fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"RGB histograms saved to: {save_path}")
            plt.show()
            
        except Exception as e:
            print(f"Error generating RGB histograms: {e}")


def main_menu():
    """Display the main menu and handle user input."""
    processor = ImageProcessor()
    
    while True:
        print("\n" + "="*50)
        print("    DIGITAL IMAGE PROCESSING SUITE")
        print("="*50)
        print("1. Load and Display Image")
        print("2. Encrypt Image")
        print("3. Decrypt Image")
        print("4. Apply Wiener Filter")
        print("5. Generate Grayscale Histogram")
        print("6. Generate RGB Histograms")
        print("7. Process All Operations")
        print("8. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-8): ").strip()
        
        try:
            if choice == '1':
                image = processor.load_image()
                if image is not None:
                    processor.display_image(image, "Original Image")
                    
            elif choice == '2':
                if processor.current_image is None:
                    print("Please load an image first!")
                    continue
                encrypted = processor.encrypt_image(processor.current_image)
                processor.display_image(encrypted, "Encrypted Image")
                
            elif choice == '3':
                if processor.encryption_key is None:
                    print("Please encrypt an image first!")
                    continue
                # Assuming we have the encrypted image
                encrypted = np.bitwise_xor(processor.current_image, processor.encryption_key)
                decrypted = processor.decrypt_image(encrypted)
                processor.display_image(decrypted, "Decrypted Image")
                
            elif choice == '4':
                filtered = processor.apply_wiener_filter()
                if filtered is not None:
                    processor.display_image(filtered, "Wiener Filtered Image")
                    
            elif choice == '5':
                processor.plot_grayscale_histogram()
                
            elif choice == '6':
                processor.plot_rgb_histograms()
                
            elif choice == '7':
                print("Processing all operations...")
                # Load image
                image = processor.load_image()
                if image is None:
                    continue
                    
                # Display original
                processor.display_image(image, "Original Image")
                
                # Encrypt and display
                encrypted = processor.encrypt_image(image)
                processor.display_image(encrypted, "Encrypted Image")
                
                # Decrypt and display
                decrypted = processor.decrypt_image(encrypted)
                processor.display_image(decrypted, "Decrypted Image")
                
                # Apply Wiener filter
                filtered = processor.apply_wiener_filter()
                if filtered is not None:
                    processor.display_image(filtered, "Wiener Filtered Image")
                
                # Generate histograms
                processor.plot_grayscale_histogram()
                processor.plot_rgb_histograms()
                
                print("All operations completed!")
                
            elif choice == '8':
                print("Thank you for using the Digital Image Processing Suite!")
                sys.exit(0)
                
            else:
                print("Invalid choice! Please enter a number between 1-8.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Starting Digital Image Processing Suite...")
    main_menu()

#!/usr/bin/env python3
"""
Digital Image Processing Suite
==============================

A comprehensive image processing toolkit that includes:
- Image encryption/decryption using XOR operations
- Wiener filtering for noise reduction
- Histogram analysis (grayscale and RGB)
- Interactive image processing pipeline

Author: Your Name
Date: October 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.signal import wiener
import imageio
import os
import sys
from typing import Optional, Tuple


class ImageProcessor:
    """Main class for image processing operations."""
    
    def __init__(self, default_image_path: str = "image.png"):
        """
        Initialize the ImageProcessor.
        
        Args:
            default_image_path: Default path to use when no image is specified
        """
        self.default_image_path = default_image_path
        self.current_image = None
        self.encryption_key = None
        
    def load_image(self, path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Load an image with error handling and ensure RGB mode.
        
        Args:
            path: Path to the image file. If None, prompts user for input.
            
        Returns:
            numpy array of the image or None if loading failed
        """
        if path is None:
            path = input("Enter the image path (press Enter for default): ").strip()
            
        if not path:
            path = self.default_image_path
            
        if not os.path.exists(path):
            print(f"Error: File '{path}' not found!")
            return None
            
        try:
            image = np.array(Image.open(path).convert("RGB"))
            self.current_image = image
            print(f"Successfully loaded image: {path}")
            print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} pixels")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def display_image(self, image: np.ndarray, title: str = "Image", save_path: Optional[str] = None):
        """
        Display an image using matplotlib.
        
        Args:
            image: Image array to display
            title: Title for the plot
            save_path: Optional path to save the displayed image
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(title, fontsize=16)
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Image saved to: {save_path}")
            
        plt.show()
    
    def generate_encryption_key(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Generate a random encryption key for XOR operations.
        
        Args:
            shape: Shape of the image (height, width, channels)
            
        Returns:
            Random key array
        """
        np.random.seed(42)  # For reproducible results
        key = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        self.encryption_key = key
        return key
    
    def encrypt_image(self, image: np.ndarray, save_path: str = "encrypted_image.png") -> np.ndarray:
        """
        Encrypt an image using XOR operation.
        
        Args:
            image: Input image array
            save_path: Path to save the encrypted image
            
        Returns:
            Encrypted image array
        """
        image = image.astype(np.uint8)
        key = self.generate_encryption_key(image.shape)
        
        encrypted = np.bitwise_xor(image, key)
        
        # Save encrypted image
        encrypted_img = Image.fromarray(encrypted)
        encrypted_img.save(save_path)
        print(f"Encrypted image saved to: {save_path}")
        
        return encrypted
    
    def decrypt_image(self, encrypted_image: np.ndarray, save_path: str = "decrypted_image.png") -> np.ndarray:
        """
        Decrypt an image using XOR operation.
        
        Args:
            encrypted_image: Encrypted image array
            save_path: Path to save the decrypted image
            
        Returns:
            Decrypted image array
        """
        if self.encryption_key is None:
            raise ValueError("No encryption key available. Encrypt an image first.")
        
        decrypted = np.bitwise_xor(encrypted_image, self.encryption_key)
        
        # Save decrypted image
        decrypted_img = Image.fromarray(decrypted)
        decrypted_img.save(save_path)
        print(f"Decrypted image saved to: {save_path}")
        
        return decrypted
    
    def apply_wiener_filter(self, image_path: Optional[str] = None, save_path: str = "wiener_filtered.png") -> Optional[np.ndarray]:
        """
        Apply Wiener filter for noise reduction.
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the filtered image
            
        Returns:
            Filtered image array or None if failed
        """
        if image_path is None:
            image_path = input("Enter path for Wiener filtering (press Enter for default): ").strip()
            
        if not image_path:
            image_path = self.default_image_path
            
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found!")
            return None
        
        try:
            # Load image using imageio
            image = imageio.imread(image_path)
            
            # Apply Wiener filter
            filtered_image = wiener(image)
            
            # Save filtered image
            imageio.imwrite(save_path, filtered_image.astype(np.uint8))
            print(f"Wiener filtered image saved to: {save_path}")
            
            return filtered_image
        except Exception as e:
            print(f"Error applying Wiener filter: {e}")
            return None
    
    def plot_grayscale_histogram(self, image_path: Optional[str] = None, save_path: str = "grayscale_histogram.png"):
        """
        Plot histogram of a grayscale image.
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the histogram plot
        """
        if image_path is None:
            image_path = input("Enter path for grayscale histogram (press Enter for default): ").strip()
            
        if not image_path:
            image_path = self.default_image_path
            
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found!")
            return
        
        try:
            # Load image as grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print("Error: Could not read the image in grayscale!")
                return
            
            # Calculate histogram
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            
            # Plot histogram
            plt.figure(figsize=(12, 6))
            plt.title("Grayscale Histogram", fontsize=16)
            plt.xlabel("Gray Level", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.bar(range(256), hist.ravel(), color="black", alpha=0.7)
            plt.grid(True, alpha=0.3)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Grayscale histogram saved to: {save_path}")
            plt.show()
            
        except Exception as e:
            print(f"Error generating grayscale histogram: {e}")
    
    def plot_rgb_histograms(self, image_path: Optional[str] = None, save_path: str = "rgb_histograms.png"):
        """
        Plot RGB component histograms.
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the histogram plots
        """
        if image_path is None:
            image_path = input("Enter path for RGB histogram (press Enter for default): ").strip()
            
        if not image_path:
            image_path = self.default_image_path
            
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found!")
            return
        
        try:
            # Open image and convert to RGB
            img = Image.open(image_path).convert("RGB")
            r, g, b = img.split()
            
            plt.figure(figsize=(15, 5))
            colors = ["red", "green", "blue"]
            channels = [r, g, b]
            titles = ["Red Component Histogram", "Green Component Histogram", "Blue Component Histogram"]
            
            for i, (channel, color, title) in enumerate(zip(channels, colors, titles)):
                plt.subplot(1, 3, i + 1)
                plt.hist(np.array(channel).ravel(), bins=256, color=color, alpha=0.6, edgecolor='black')
                plt.title(title, fontsize=14)
                plt.xlabel("Pixel Value", fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"RGB histograms saved to: {save_path}")
            plt.show()
            
        except Exception as e:
            print(f"Error generating RGB histograms: {e}")


def main_menu():
    """Display the main menu and handle user input."""
    processor = ImageProcessor()
    
    while True:
        print("\n" + "="*50)
        print("    DIGITAL IMAGE PROCESSING SUITE")
        print("="*50)
        print("1. Load and Display Image")
        print("2. Encrypt Image")
        print("3. Decrypt Image")
        print("4. Apply Wiener Filter")
        print("5. Generate Grayscale Histogram")
        print("6. Generate RGB Histograms")
        print("7. Process All Operations")
        print("8. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-8): ").strip()
        
        try:
            if choice == '1':
                image = processor.load_image()
                if image is not None:
                    processor.display_image(image, "Original Image")
                    
            elif choice == '2':
                if processor.current_image is None:
                    print("Please load an image first!")
                    continue
                encrypted = processor.encrypt_image(processor.current_image)
                processor.display_image(encrypted, "Encrypted Image")
                
            elif choice == '3':
                if processor.encryption_key is None:
                    print("Please encrypt an image first!")
                    continue
                # Assuming we have the encrypted image
                encrypted = np.bitwise_xor(processor.current_image, processor.encryption_key)
                decrypted = processor.decrypt_image(encrypted)
                processor.display_image(decrypted, "Decrypted Image")
                
            elif choice == '4':
                filtered = processor.apply_wiener_filter()
                if filtered is not None:
                    processor.display_image(filtered, "Wiener Filtered Image")
                    
            elif choice == '5':
                processor.plot_grayscale_histogram()
                
            elif choice == '6':
                processor.plot_rgb_histograms()
                
            elif choice == '7':
                print("Processing all operations...")
                # Load image
                image = processor.load_image()
                if image is None:
                    continue
                    
                # Display original
                processor.display_image(image, "Original Image")
                
                # Encrypt and display
                encrypted = processor.encrypt_image(image)
                processor.display_image(encrypted, "Encrypted Image")
                
                # Decrypt and display
                decrypted = processor.decrypt_image(encrypted)
                processor.display_image(decrypted, "Decrypted Image")
                
                # Apply Wiener filter
                filtered = processor.apply_wiener_filter()
                if filtered is not None:
                    processor.display_image(filtered, "Wiener Filtered Image")
                
                # Generate histograms
                processor.plot_grayscale_histogram()
                processor.plot_rgb_histograms()
                
                print("All operations completed!")
                
            elif choice == '8':
                print("Thank you for using the Digital Image Processing Suite!")
                sys.exit(0)
                
            else:
                print("Invalid choice! Please enter a number between 1-8.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Starting Digital Image Processing Suite...")
    main_menu()