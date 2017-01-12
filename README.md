# [WIP] Data augmentation tools

This repository provides tools for data augmentation using OpenCV.

## Usage

```python
cat = cv2.imread('cat.jpg')
dog = cv2.imread('dog.jpg')

augmented_images = data_augmentation([cat, dog])

for i in range(len(augmented_images)):
	cv2.imwrite('results/' + str(i)+'.jpg', augmented_images[i])
```

## Data augmentation techniques

- [x] Inversion
- [x] Sobel derivative
- [x] Scharr derivative
- [x] Laplacian
- [x] Blur
- [x] Gaussian blur
- [x] Median blur
- [x] Bilateral blur
- [x] Horizontal flips
- [ ] Distortions
- [ ] Random scales
- [ ] Stretching
- [ ] Color jittering
- [ ] Shearing

