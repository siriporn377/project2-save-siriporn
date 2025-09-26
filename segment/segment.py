import os
import cv2
import numpy as np
from rembg import remove
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb
from skimage.util import img_as_float
from sklearn.cluster import KMeans

# Path Setup 
base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, 'photo', 'input36.jpg')
output_dir = os.path.join(base_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# Load & Remove Background 
img = cv2.imread(input_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_no_bg = remove(img_rgb)
img_no_bg_rgb = img_no_bg[:, :, :3]
# Save the background-removed image (for presentation) 
output_removed_bg_path = os.path.join(output_dir, 'input36_removed_bg.png')
cv2.imwrite(output_removed_bg_path, cv2.cvtColor(img_no_bg_rgb, cv2.COLOR_RGB2BGR))
print("✔️ Background-removed image saved at:", output_removed_bg_path)


# Smooth image before segmentation 
blurred = cv2.bilateralFilter(img_no_bg_rgb, d=9, sigmaColor=75, sigmaSpace=75)
image_float = img_as_float(blurred)

#  Felzenszwalb segmentation 
segments = felzenszwalb(image_float, scale=100, sigma=0.5, min_size=50)

# Create segmented image (average color per segment) 
segmented_image = label2rgb(segments, image_float, kind='avg') #label map
segmented_img_uint8 = (segmented_image * 255).astype(np.uint8)
cv2.imwrite(os.path.join(output_dir, 'input36_output+kmean.jpg'), cv2.cvtColor(segmented_img_uint8, cv2.COLOR_RGB2BGR))
np.save(os.path.join(output_dir, 'labels36.npy'), segments)

#  Apply K-means on segmented image to reduce to 10 colors 
H, W, _ = segmented_img_uint8.shape
pixels = segmented_img_uint8.reshape((-1, 3))
k = 10
kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
labels_k = kmeans.labels_#บอกว่าสีของแต่ละพิกเซลอยู่ในกลุ่มไหน
centers_k = kmeans.cluster_centers_.astype(np.uint8)#ค่าสี RGB ของแต่ละกลุ่มหลังจัดกลุ่มแล้ว

#  Create subfolder for K-means layers 
kmeans_layer_dir = os.path.join(output_dir, 'kmeans_layers36')
os.makedirs(kmeans_layer_dir, exist_ok=True)

#  Save each layer of color (K-means layer) 
for i in range(k):
    mask = (labels_k.reshape((H, W)) == i)#คืนกลับให้เป็นภาพ 2 มิติ ขนาดเท่าภาพต้นฉบับ
    layer = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background
    layer[mask] = centers_k[i]
    layer_bgr = cv2.cvtColor(layer, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(kmeans_layer_dir, f'layer_kmeans_{i}.jpg'), layer_bgr) #บันทึกภาพแต่ละ layerลงไฟล์

print("✔️ All layers saved in:", kmeans_layer_dir)
