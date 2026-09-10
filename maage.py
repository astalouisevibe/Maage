import cv2
from ultralytics import YOLO

# prætrænet YOLOv8-model (nano-versionen 'n' er lynhurtig, 'm' eller 'l' er mere præcise)
model = YOLO("yolov8n.pt") 

# 2. pt sat til webcam på computeren, kan erstattes med ekstern kamera
camera_source = 0 
cap = cv2.VideoCapture(camera_source)

if not cap.isOpened():
    print("Fejl: Kunne ikke åbne kamera-streamen.")
    exit()

print("Programmet kører... Tryk på 'q' for at afslutte.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Kør maskinlæringsmodellen på det aktuelle kamerabillede
    # filtrerer på classes=[14], som er "bird" i COCO-datasættet
    results = model(frame, classes=[14], conf=0.4, verbose=False)

    # 4. Loop igennem de fundne fugle/måger
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Få koordinaterne for boksen omkring mågen (X1, Y1 er top-venstre, X2, Y2 er bund-højre)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Få modellens sikkerhedsscore (f.eks. 0.85 = 85% sikker)
            confidence = float(box.conf[0])

            print(f"MÅGE DETEKTERET! Placering i billedet: Top-venstre: ({x1}, {y1}), Bund-højre: ({x2}, {y2}) | Sikkerhed: {confidence:.2f}")

            # Tegn en grøn boks omkring mågen på videoskærmen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Maage: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Vis live-videoen med bokse på skærmen (kan deaktiveres hvis det kører i baggrunden på en server)
    cv2.imshow("Tagovervaagning - Maager", frame)

    # Stop hvis der trykkes på 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
