# Data augmentation tools [WIP]

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

- [x] hello
- [] hfsdh

