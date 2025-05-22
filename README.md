# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

## Program:

## Name: Jai Surya R
## Reg.No: 212223230084

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Images ---
# Load the face image
img = cv2.imread('/content/JAI_SURYA_PROFILE-_1_.jpeg') # Ensure this path is correct for your face image
if img is None:
    print("Error: Could not load face image. Please check the path and if the file exists.")
    exit()

# Load the Sunglass image with Alpha channel
glassPNG = cv2.imread('/content/sunglass.png', -1) # -1 ensures alpha channel is loaded
if glassPNG is None:
    print("Error: Could not load sunglass image. Please check the path and if the file exists.")
    exit()


# --- 2. Initialize for Augmentation and Load Cascades ---
faceWithGlassesArithmetic = img.copy() # This will be the image we modify

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty():
    print("Error: Could not load face cascade XML. Check path.")
    exit()
if eye_cascade.empty():
    print("Error: Could not load eye cascade XML. Check path.")
    exit()

gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 3. Detect Faces and Eyes ---
faces = face_cascade.detectMultiScale(gray_face, 1.3, 5)

print(f"Detected {len(faces)} face(s).")

for (x, y, w, h) in faces:
    roi_gray = gray_face[y:y+h, x:x+w]
    roi_color = faceWithGlassesArithmetic[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)

    print(f"Detected {len(eyes)} eye(s) within the face ROI.")

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0]) # Sort by x-coordinate for left/right

        (ex1, ey1, ew1, eh1) = eyes[0] # Left eye
        (ex2, ey2, ew2, eh2) = eyes[1] # Right eye

        left_eye_center = (ex1 + ew1 // 2, ey1 + eh1 // 2)
        right_eye_center = (ex2 + ew2 // 2, ey2 + eh2 // 2)

        # --- 4. Dynamic Sunglass Sizing and Placement (ADJUSTMENTS HERE) ---

        # Calculate the distance between eye centers
        eye_distance = np.sqrt((right_eye_center[0] - left_eye_center[0])**2 +
                               (right_eye_center[1] - left_eye_center[1])**2)

        glass_width = int(eye_distance * 2.2) # INCREASE THIS MULTIPLIER for wider glasses

        # Ensure a minimum width to prevent issues
        if glass_width < 100:
            glass_width = 100

        original_glass_aspect_ratio = glassPNG.shape[1] / glassPNG.shape[0]
        glass_height = int(glass_width / original_glass_aspect_ratio)

        print(f"Calculated glass dimensions: {glass_width}x{glass_height}")

        resized_glassPNG = cv2.resize(glassPNG, (glass_width, glass_height))
        if resized_glassPNG.shape[-1] == 3:
            resized_glassPNG = cv2.cvtColor(resized_glassPNG, cv2.COLOR_BGR2BGRA)

        glassBGR = resized_glassPNG[:, :, :3]
        glassMask1 = resized_glassPNG[:, :, 3]

        center_between_eyes_x = (left_eye_center[0] + right_eye_center[0]) // 2

        glass_start_x = center_between_eyes_x - (glass_width // 2)

        glass_start_y = int(left_eye_center[1] - glass_height * 0.3) # ADJUST THIS MULTIPLIER for vertical position


        # Ensure coordinates are within bounds
        x_offset = max(0, glass_start_x)
        y_offset = max(0, glass_start_y)

        x_end_roi = min(roi_color.shape[1], x_offset + glass_width)
        y_end_roi = min(roi_color.shape[0], y_offset + glass_height)

        eyeROI_actual = roi_color[y_offset:y_end_roi, x_offset:x_end_roi]

        current_glass_width = eyeROI_actual.shape[1]
        current_glass_height = eyeROI_actual.shape[0]

        glassBGR_resized_to_roi = cv2.resize(glassBGR, (current_glass_width, current_glass_height))
        glassMask1_resized_to_roi = cv2.resize(glassMask1, (current_glass_width, current_glass_height))

        # --- 5. Blending ---
        glassMask = cv2.merge((glassMask1_resized_to_roi, glassMask1_resized_to_roi, glassMask1_resized_to_roi))
        glassMask = glassMask.astype(np.float32) / 255

        eyeROI_actual = eyeROI_actual.astype(np.float32) / 255
        glassBGR_resized_to_roi = glassBGR_resized_to_roi.astype(np.float32) / 255

        maskedEye = cv2.multiply(eyeROI_actual, (1 - glassMask))
        maskedGlass = cv2.multiply(glassBGR_resized_to_roi, glassMask)
        eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

        eyeRoiFinal = (eyeRoiFinal * 255).astype(np.uint8)

        roi_color[y_offset:y_end_roi, x_offset:x_end_roi] = eyeRoiFinal

        break # Process only the first detected face

    else:
        print("Did not detect enough eyes in the face ROI to place glasses for this face.")

glassPNG_for_display = cv2.imread('/content/sunglass.png', -1)
if glassPNG_for_display is None:
    glassPNG_for_display = np.zeros((100, 100, 4), dtype=np.uint8) # Placeholder if loading fails

if glassPNG_for_display.shape[-1] == 3:
    glassPNG_for_display = cv2.cvtColor(glassPNG_for_display, cv2.COLOR_BGR2BGRA)

display_glass_color = cv2.resize(glassPNG_for_display[:, :, :3], (180, 80))
display_glass_mask = cv2.resize(glassPNG_for_display[:, :, 3], (180, 80))


plt.figure(figsize=[15, 6])
plt.subplot(131)
plt.imshow(display_glass_color[:,:,::-1])
plt.title('Sunglass Color channels')
plt.axis('off')

plt.subplot(132)
plt.imshow(display_glass_mask, cmap='gray')
plt.title('Sunglass Alpha channel')
plt.axis('off')

if 'eyeRoiFinal' in locals():
    plt.subplot(133)
    plt.imshow(eyeRoiFinal[:,:,::-1])
    plt.title('Augmented Eye and Sunglass')
    plt.axis('off')
else:
    plt.subplot(133)
    plt.text(0.5, 0.5, "No Sunglasses placed\n(Face/Eyes not detected)", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, color='red')
    plt.title('Augmented Eye and Sunglass')
    plt.axis('off')

plt.tight_layout()
plt.show()


# Display masked and blended regions (if glasses were placed)
if 'maskedEye' in locals() and 'maskedGlass' in locals() and 'eyeRoiFinal' in locals():
    plt.figure(figsize=[20, 8])
    plt.subplot(131)
    # Ensure float images are displayed correctly; they are already in 0-1 range for matplotlib
    plt.imshow(maskedEye[...,::-1])
    plt.title("Masked Eye Region")
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(maskedGlass[...,::-1])
    plt.title("Masked Sunglass Region")
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(eyeRoiFinal[...,::-1])
    plt.title("Augmented Eye and Sunglass (Final Blended ROI)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping display of masked/blended regions as no sunglasses were placed.")


# Final result with blending
plt.figure(figsize=[15, 8])
plt.subplot(121)
plt.imshow(img[:,:,::-1])
plt.title("Original Image")
plt.axis('off')

plt.subplot(122)
plt.imshow(faceWithGlassesArithmetic[:,:,::-1])
plt.title("Image With Sunglasses")
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nScript finished. Check the displayed plots for the results.")
```

## Output:
![image](https://github.com/user-attachments/assets/ecc84372-c4f0-4aa6-aa33-e284d212d72d)

![image](https://github.com/user-attachments/assets/e08bd699-1304-4e22-b3b7-629c16fcf814)




##  Result:
Thus, the creative project designed to overlay sunglasses on individual passport size photo has been successfully executed.


