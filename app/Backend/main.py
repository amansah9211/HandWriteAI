from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import json
import io
import cv2
from PIL import Image

app = FastAPI(title="HandWriteAI API")

model = tf.keras.models.load_model("emnist_cnn.keras")

with open("result.json", "r", encoding="utf-8") as file:
    class_mapping = json.load(file)


@app.get("/")
def root():
    return {
        "message": "HandWriteAI Backend is running"
    }

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img_array = np.array(img)

    _, binary = cv2.threshold(
        img_array,
        100,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "No character detected"
        )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )


    character_contour = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= 10 and h >= 10:
            character_contour = contour
            break

    if character_contour is None:
        raise ValueError(
            "Character not detected"
        )

    x, y, w, h = cv2.boundingRect(
        character_contour
    )

    cropped = binary[y:y+h,x:x+w]

    print("Original character size :",w,"x",h)

    max_size = 20
    scale = min(max_size / w,max_size / h)

    new_width = max(1,int(w * scale))

    new_height = max(1,int(h * scale))

    cropped = cv2.resize(
        cropped,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


    print("Resized character size  :",new_width,"x",new_height)
    canvas = np.zeros((28, 28),dtype=np.uint8)

    start_x = (28 - new_width) // 2
    start_y = (28 - new_height) // 2


    canvas[start_y:start_y + new_height,start_x:start_x + new_width] = cropped

    kernel = np.ones((2, 2),np.uint8)

    canvas = cv2.dilate(canvas,kernel,iterations=2)

    print("Canvas size             :",canvas.shape)
    print("Character start position:",(start_x, start_y))
    print("Non-zero pixels         :",np.count_nonzero(canvas))
    print("Mean pixel value        :",round(float(canvas.mean()),2))
    print("Maximum pixel value     :",canvas.max())
    print("Minimum pixel value     :",canvas.min())

    cv2.imwrite("debug_processed.png",canvas)
    print("Debug image saved as: debug_processed.png")

    img = canvas.astype("float32") / 255.0
    img = img.reshape(1,28,28,1)

    print("Final CNN input shape:",img.shape)
    print("Input min:",img.min())
    print("Input max:",img.max())
    return img

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:
        img_data = await file.read()
        img = preprocess_image(img_data)

        prediction = model.predict(img,verbose=0)[0]
        top5_indices = np.argsort(prediction)[-5:][::-1]
        top5 = []

        for idx in top5_indices:
            class_id = int(idx)
            top5.append({
                "class": class_id,
                "character":
                    class_mapping[
                        str(class_id)
                    ],

                "confidence":
                    round(
                        float(
                            prediction[idx]
                        ) * 100,
                        2
                    )
            })

        class_id = int(np.argmax(prediction))
        confidence = (float(prediction[class_id]) * 100)
        character = class_mapping[str(class_id)]

        return {
            "class": class_id,
            "character": character,
            "confidence": round(confidence,2),
            "top5": top5
        }


    except ValueError as e:
        return {
            "error": str(e)
        }


    except Exception as e:
        return {
            "error": str(e)
        }