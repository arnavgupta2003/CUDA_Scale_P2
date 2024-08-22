import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pathlib import Path

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def edge_detection(image):
    # Convert image to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Allocate device memory and transfer data
    img_gpu = cuda.mem_alloc(gray.nbytes)
    cuda.memcpy_htod(img_gpu, gray)
    
    # Define CUDA kernel
    mod = SourceModule("""
    __global__ void edge_detection_kernel(unsigned char *img, int width, int height) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = y * width + x;
        if (x < width && y < height) {
            // Apply a simple edge detection kernel (e.g., Sobel)
            // For simplicity, just set it to zero (as an example)
            img[idx] = 0;
        }
    }
    """)
    
    kernel = mod.get_function("edge_detection_kernel")
    
    # Launch kernel
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(gray.shape[1] / block_size[0])), int(np.ceil(gray.shape[0] / block_size[1])), 1)
    kernel(img_gpu, np.int32(gray.shape[1]), np.int32(gray.shape[0]), block=block_size, grid=grid_size)
    
    # Retrieve result from GPU
    result = np.empty_like(gray)
    cuda.memcpy_dtoh(result, img_gpu)
    
    return result

def main(input_image_path, output_image_path):
    image = load_image(input_image_path)
    edges = edge_detection(image)
    save_image(edges, output_image_path)

if __name__ == "__main__":
    data_dir = Path('../data')
    input_image_path = data_dir / 'input_image.tiff'
    output_image_path = data_dir / 'output_image.png'
    main(str(input_image_path), str(output_image_path))
